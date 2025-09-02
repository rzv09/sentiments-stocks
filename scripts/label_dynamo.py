import argparse
import joblib
import boto3
from boto3.dynamodb.conditions import Attr

# --- TODO 1: Load the pipeline ---
def load_pipeline(path: str):
    """
    TODO:
    - joblib.load(path) 
    - return pipeline (use ["pipeline"] if you saved it in a dict)
    """
    pass


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
    pass


# --- TODO 3: Predict sentiment ---
def predict_batch(pipeline, texts: list[str]) -> list[dict]:
    """
    TODO:
    - Use pipeline.predict and pipeline.predict_proba
    - proba[:, 1] = positive probability
    - Map 1 -> "positive", 0 -> "negative"
    - Return list of dicts: { "label": str, "confidence": float }
    """
    pass


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
    pass


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
        try:
            update_item(table, url_hash, pred["label"], float(pred["confidence"]))
            updated += 1
        except Exception as e:
            # If duplicate -> count as skipped
            if "ConditionalCheckFailedException" in str(e):
                skipped += 1
            else:
                raise

    print(f"Finished: updated={updated}, skipped={skipped}, total={len(items)}")
    

if __name__ == "__main__":
    main()
