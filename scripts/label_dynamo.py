import argparse
import joblib
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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



# --- TODO 3: Predict sentiment ---
def predict_batch(pipeline, texts: list[str]) -> list[dict]:
    """
    TODO:
    - Use pipeline.predict and pipeline.predict_proba
    - proba[:, 1] = positive probability
    - Map 1 -> "positive", 0 -> "negative"
    - Return list of dicts: { "label": str, "confidence": float }
    """
    label_map = {
        1: "positive",
        0: "negative"
    }
    preds = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)[:, 1]

    return [
        {"label": label_map[p], "confidence": float(prob)}
        for p, prob in zip(preds, probas)
    ]


# --- TODO 4: Update DynamoDB ---
def update_item(table, url_hash: str, sentiment: str, confidence: float):
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
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            print("Conditional check failed. Item already has sentiment field.")
            return False
        else:
            raise e


# --- Main driver ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="ml/models/baseline/tfidf_logreg.joblib")
    parser.add_argument("--table", default="SentimentsStocksRawNews")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    # DynamoDB table handle
    ddb = boto3.resource("dynamodb")
    table = ddb.Table(args.table)

    # Load pipeline
    pipeline = load_pipeline(args.artifact)

    # Fetch items
    items = fetch_unlabeled(table, args.limit)

    updated = 0
    skipped = 0

    for item in items:
        text = item.get("headline", "")
        url_hash = item["url_hash"]

        if not text.strip():
            skipped += 1
            continue

        pred = predict_batch(pipeline, [text])[0]

        if update_item(table, url_hash, pred["label"], float(pred["confidence"])):
            updated += 1
        else:
            skipped += 1

    print(f"Finished: updated={updated}, skipped={skipped}, total={len(items)}")
    

if __name__ == "__main__":
    main()
