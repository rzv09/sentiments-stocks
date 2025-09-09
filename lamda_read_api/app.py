import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import logging
from typing import Any, Dict, List, Tuple, Optional
import os
import json
from datetime import datetime, timezone, timedelta
"""
The lambda is exposed through AWS API Gateway. 
The webapp can communicate with the Lambda through this API Gateway.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True
)
log = logging.getLogger(__name__)

TABLE_NAME = os.getenv("TABLE_NAME", "SentimentsStocksRawNews")
INDEX_NAME = os.getenv("INDEX_NAME", "")
if not INDEX_NAME:
    raise RuntimeError("INDEX_NAME env var is required")

DEFAULT_WINDOW_HOURS = os.getenv("DEFAULT_WINDOW_HOURS", 24)
MAX_LIMIT = os.getenv("MAX_LIMIT", 100)

_session = boto3.Session()
_ddb = _session.resource("dynamodb")
_table = _ddb.Table(TABLE_NAME)

def _ok(body: Dict[str, Any], status: int = 200) -> Dict[str, Any]:
    """return API Gateway compatible response"""

    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,OPTIONS"
        },
        "body": json.dumps(body)
    }

def _err(message: str, status: int = 400) -> Dict[str, Any]:
    "return error response helper with same headers as _ok"

    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,OPTIONS"
        },
        "body": json.dumps({"error": message})
    }

def _parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and sanitize query params from API Gateway"""
    
    payload = event.get("queryStringParameters") or {}
    
    ticker = payload.get("ticker")
    if not ticker:
        raise ValueError("ticker is required")
    ticker_normalized = ticker.strip().upper()
    
    hours = int(payload.get("hours", DEFAULT_WINDOW_HOURS))
    limit = int(payload.get("limit", MAX_LIMIT))

    min_conf_raw = payload.get("min_conf")
    min_conf = None
    if min_conf_raw is not None:
        try:
            min_conf = float(min_conf_raw)
            min_conf = max(0.0, min(min_conf, 1.0))
        except ValueError:
            min_conf = None

    

    return {"ticker": ticker_normalized,
                       "hours": hours,
                       "limit": limit,
                       "min_conf": min_conf}

def _time_window(hours: int) -> Tuple[str, str]:
    """
    Compute ISO 8601 UTC start/end strings for the window.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    return (start.isoformat(), now.isoformat())

def _query_news(
    ticker: str,
    start_iso: str,
    end_iso: str,
    limit: int,
    last_evaluated_key: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    TODO: Query the GSI for a given ticker and time range.
    - Use table.query with:
        IndexName=INDEX_NAME
        KeyConditionExpression=Key("ticker").eq(ticker) & Key("published_utc").between(start_iso, end_iso)
        Limit=limit
        ScanIndexForward=False  # newest first
        ExclusiveStartKey=last_evaluated_key (if provided)
    - Return (items, new_last_evaluated_key)
    """
    kwargs = {
        "IndexName": INDEX_NAME,
        "KeyConditionExpression":Key("ticker").eq(ticker) & Key("published_utc").between(start_iso, end_iso),
        "Limit": limit,
        "ScanIndexForward":False,
        "ExclusiveStartKey":last_evaluated_key   
    }
    try:
        response = _table.query(**kwargs)
        items = response["Items"]
        last_evaluated_key = response["LastEvaluatedKey"] or None
        return items, last_evaluated_key
    except ClientError as e:
        log.error("DynamoDB query failed", extra={"error": str(e), "kwargs": kwargs})
        raise

def query_all_news(ticker, start_iso, end_iso, page_limit=200, max_items=2000):
    items, lek, out = [], None, []
    while True:
        # fetch one "page"
        items, lek = _query_news(ticker, start_iso, end_iso, page_limit, lek)
        out.extend(items)

        # stop if no more pages OR we hit our max_items guard
        if not lek or len(out) >= max_items:
            break
    return out[:max_items]


def _partition_sentiment(
    items: List[Dict[str, Any]],
    min_conf: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    TODO: Split items into 'positive' and 'negative' lists.
    - For each item:
        label = item.get("sentiment")  # may be missing if unlabeled; skip those
        conf = float(item.get("confidence", 0))
        if min_conf is not None and conf < min_conf: skip
        Add compact dicts (headline, url, published_utc, confidence) to the bucket
    - Sort each bucket by published_utc desc (if needed; query already returns sorted)
    - Return {"positive": [...], "negative": [...]}
    """
    pos_bucket: List[Dict[str, Any]] = []
    neg_bucket: List[Dict[str, Any]] = []

    for item in items:
        label = item.get("sentiment")
        if not label:
            continue
        conf = float(item.get("confidence", 0))
        if min_conf is not None and conf < min_conf:
            continue
        compact = {
            "headline": item.get("headline"),
            "url": item.get("url"),
            "published_utc": item.get("published_utc"),
            "confidence": conf
        }
        if label == "positive":
            pos_bucket.append(compact)
        else:
            neg_bucket.append(compact)
        
    pos_bucket = sorted(pos_bucket, key=lambda x: x.get("published_utc", ""), reverse=True)
    neg_bucket = sorted(neg_bucket, key=lambda x: x.get("published_utc", ""), reverse=True)
    
    return {"positive": pos_bucket, "negative": neg_bucket}


def _paginate_token_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    TODO (optional): Accept a client-provided pagination token.
    - Read query param 'next' (base64-encoded JSON) and decode to dict
    - Return dict for ExclusiveStartKey or None
    """
    pass

def _encode_pagination_token(last_key: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    TODO (optional): Encode DynamoDB LastEvaluatedKey as a base64 string for clients.
    - If last_key is None: return None
    - Else: json.dumps(last_key).encode -> base64 urlsafe encode -> str
    """
    pass

# ---------- handler ----------
def handler(event, context):
    """
    TODO:
    - Parse params via _parse_event; validate presence of ticker.
    - Compute time window via _time_window.
    - (Optional) Read pagination token via _paginate_token_from_event.
    - Call _query_news with limit; get items + last_key.
    - Partition by sentiment via _partition_sentiment (respect min_conf).
    - Build response payload: ticker, window_hours, counts, positive[], negative[], next (if any).
    - Return with _ok(...).
    - On exceptions: log.exception(...) and return _err("...").
    """
    try:
        params = _parse_event(event)
        start_t, end_t = _time_window(params["hours"])
        news = query_all_news(params["ticker"], start_t, end_t)
        partitioned_news = _partition_sentiment(news, params["min_conf"])
        response_payload = {
            "ticker": params["ticker"],
            "window_hours": params["hours"],
            "count_positive": len(partitioned_news["positive"]),
            "count_negative": len(partitioned_news["negative"]),
            "positive": partitioned_news["positive"],
            "negative": partitioned_news["negative"]
            
        }

        return _ok(response_payload)

    except Exception as e:
        log.exception("Could not retrieve ticker data from DynamoDB using specified params. See exception: %s", str(e))
        return _err("Could not retrieve ticker data from DynamoDB using specified params. See exception: %s", str(e))




