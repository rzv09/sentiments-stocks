import argparse
import joblib
import logging
import boto3
import os
import types
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
TABLE_NAME = os.getenv("TABLE_NAME", "SentimentsStocksRawNews")

BATCH_LIMIT = os.getenv("BATCH_LIMIT", "100")
THRESHOLD = os.getenv("THRESHOLD", "0.5")
MODEL_PATH = os.getenv("MODEL_PATH", "ml/models/baseline/tfidf_logreg.joblib")
# Stamped onto every row this labeler writes, so a row is always attributable
# to the model that produced it.
MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-0.2-3class")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ddb = boto3.resource("dynamodb")
table = ddb.Table(TABLE_NAME)

_PIPELINE = joblib.load(Path(MODEL_PATH))


def _verify_model_version(model_path: str, expected: str) -> None:
    """Fail cold start if the shipped artifact is not the model we claim to be.

    Every row this function writes is stamped with MODEL_VERSION, and consumers
    branch on that stamp to decide how to read the score. Shipping a stale
    artifact under a new version string is therefore worse than shipping nothing:
    it mislabels data silently, and downstream code trusts the label. Refuse.
    """
    marker = Path(model_path).parent / "MODEL_VERSION"
    if not marker.exists():
        log.warning("No MODEL_VERSION beside %s; cannot verify the artifact is %s",
                    model_path, expected)
        return
    found = marker.read_text().strip()
    if found != expected:
        raise RuntimeError(
            f"Model artifact is '{found}' but this labeler stamps rows as "
            f"'{expected}'. Rebuild the image after running "
            f"scripts/train_baseline.py, or set MODEL_VERSION to match.")
    log.info("Model version verified: %s", found)


_verify_model_version(MODEL_PATH, MODEL_VERSION)
log.info("Model loaded from %s", MODEL_PATH)

# n_classes tells us which contract this artifact actually honours.
log.info("Model classes: %s", getattr(_PIPELINE, "classes_", "unknown"))

# --- TODO 1: Load the pipeline ---
def load_pipeline(path: str):
    """
    TODO:
    - joblib.load(path) 
    - return pipeline (use ["pipeline"] if you saved it in a dict)
    """
    pipeline_path = Path(path)
    return joblib.load(path)


# --- TODO 2: Fetch unlabeled items ---
def fetch_unlabeled(table, limit: int):
    """
    TODO:
    - Use table.scan with a FilterExpression:
        Attr("sentiment").not_exists()
    - Limit the number of items (use Limit=limit)
    - Handle pagination via LastEvaluatedKey if needed
    - Return a list of items (each is a dict)
    """
    items = []
    scan_kwargs = {
        "FilterExpression": Attr("sentiment").not_exists(),
        "Limit": min(limit, 100)
    }
    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response["Items"])

        if "LastEvaluatedKey" not in response or len(items) >= limit:
            break # no more pages or hit our requested limit

        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    return items[:limit]


def predict_batch(pipeline, texts: list[str], threshold: float = None) -> list[dict]:
    """Score headlines with the 3-class model (baseline-0.2-3class).

    Field contract — read before changing, the wiki depends on it:

      sentiment       argmax label: positive | neutral | negative.
                      NOTE this changed meaning in v0.2. Under v0.1 it was
                      P(positive) thresholded at THRESHOLD, and could never be
                      neutral. Consumers must branch on model_version.
      confidence      Still P(positive), same DEFINITION as v0.1, and still not
                      the confidence in the predicted label. But its
                      DISTRIBUTION has moved a long way: a 3-class softmax gives
                      neutral most of the mass, so P(positive) now clusters low
                      (roughly 0.0-0.55 on real headlines) where v0.1 spanned
                      0.28-0.92 around a 0.70 split. Same meaning, different
                      scale.

                      So rows from the two models are NOT poolable: any mean,
                      baseline or drift figure computed across a mix of them is
                      wrong. Filter by model_version before aggregating. Keeping
                      the definition stable avoids a second, subtler failure -
                      a field that silently means two different things - but it
                      does not make the values interchangeable.
      sentiment_score P(positive) - P(negative), in [-1, 1]. Signed, with
                      neutral near zero. This is the field new consumers want.
      model_version   which model produced the row.

    `threshold` is accepted and ignored; a 3-class argmax has no single cutoff.
    """
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    out = []
    for row in proba:
        d = dict(zip(classes, (float(v) for v in row)))
        pos, neg = d.get("positive", 0.0), d.get("negative", 0.0)
        out.append({
            "label": max(d, key=d.get),
            "confidence": pos,
            "sentiment_score": pos - neg,
            "model_version": MODEL_VERSION,
        })
    return out


# --- TODO 4: Update DynamoDB ---
def update_item(table, url_hash: str, sentiment: str, confidence: float,
                sentiment_score: float = None, model_version: str = None) -> bool:
    """
    TODO:
    - table.update_item(
        Key={"url_hash": url_hash},
        UpdateExpression="SET sentiment = :s, confidence = :c",
        ExpressionAttributeValues={":s": sentiment, ":c": confidence},
        ConditionExpression="attribute_not_exists(sentiment)"
      )
    - Wrap in try/except:
        * If ConditionalCheckFailedException -> item already labeled
    """
    expr = "SET sentiment = :s, confidence = :c"
    vals = {":s": sentiment, ":c": Decimal(str(confidence))}
    if sentiment_score is not None:
        expr += ", sentiment_score = :ss"
        vals[":ss"] = Decimal(str(sentiment_score))
    if model_version is not None:
        expr += ", model_version = :mv"
        vals[":mv"] = model_version
    try:
        table.update_item(
            Key={"url_hash": url_hash},
            UpdateExpression=expr,
            ExpressionAttributeValues=vals,
            # Still only labels unscored items. Re-scoring the existing corpus
            # under v0.2 is a separate, deliberate backfill — not a side effect
            # of a routine run.
            ConditionExpression="attribute_not_exists(sentiment)"
        )
        log.info("Updated item with URLHash=%s; sentiment=%s; confidence=%s", url_hash,
                 sentiment, confidence)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            log.warning("Conditional check failed. Item already has sentiment field.")
            return False
        else:
            raise e


# --- Main driver ---
def handler(event, context):
    log.info("Labeler start: limit=%s threshold=%s table=%s", BATCH_LIMIT, THRESHOLD, TABLE_NAME)

    # Fetch items
    items = fetch_unlabeled(table, int(BATCH_LIMIT))

    updated = 0
    skipped = 0

    for item in items:
        text = item.get("headline", "")
        url_hash = item["url_hash"]

        if not text.strip():
            skipped += 1
            continue

        pred = predict_batch(_PIPELINE, [text], float(THRESHOLD))[0]

        if update_item(table, url_hash, pred["label"], float(pred["confidence"]),
                       pred.get("sentiment_score"), pred.get("model_version")):
            updated += 1
        else:
            skipped += 1

    log.info("Labeler done: updated=%s, skipped=%s, total=%s", updated,
             skipped, len(items))
    
    return {"updated": updated, "skipped": skipped, "total": len(items)}
    

if __name__ == "__main__":
    print(handler({}, None))
