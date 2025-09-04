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

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ddb = boto3.resource("dynamodb")
table = ddb.Table(TABLE_NAME)

_PIPELINE = joblib.load(Path(MODEL_PATH))
log.info("Model loaded from %s", MODEL_PATH)

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


def predict_batch(pipeline, texts: list[str], threshold: float) -> list[dict]:
    probas = pipeline.predict_proba(texts)[:, 1]
    labels = ["positive" if p >= threshold else "negative" for p in probas]
    return [{"label": lbl, "confidence": float(p)} for lbl, p in zip(labels, probas)]


# --- TODO 4: Update DynamoDB ---
def update_item(table, url_hash: str, sentiment: str, confidence: float) -> bool:
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
    try:
        table.update_item(
            Key={"url_hash": url_hash},
            UpdateExpression="SET sentiment = :s, confidence = :c",
            ExpressionAttributeValues={":s": sentiment, ":c": Decimal(str(confidence))},
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

        if update_item(table, url_hash, pred["label"], float(pred["confidence"])):
            updated += 1
        else:
            skipped += 1

    log.info("Labeler done: updated=%s, skipped=%s, total=%s", updated,
             skipped, len(items))
    
    return {"updated": updated, "skipped": skipped, "total": len(items)}
    

if __name__ == "__main__":
    print(handler({}, None))
