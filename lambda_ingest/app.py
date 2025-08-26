import os
import time
import hashlib
import logging
import datetime
import urllib.parse
import requests
import feedparser
import boto3
from botocore.exceptions import ClientError

# logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# env variables
table_name = os.getenv("TABLE_NAME", "SentimentsStocksRawNews")
tickers_str = os.getenv("TICKERS", "NVDA,TSM,ASML,AVGO")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

REGION = os.getenv("REGION", "US")
LANG = os.getenv("LANG", "en-US")
TIMEOUT = float(os.getenv("TIMEOUT", "10"))
RETRIES = int(os.getenv("RETRIES", "3"))

if not tickers:
    raise ValueError("No tickers provided. Set TICKERS environment variable.")

if TIMEOUT < 1 or TIMEOUT > 30:
    log.warning("TIMEOUT %s out of range (1-30). Defaulting to 10 seconds.", TIMEOUT)
    TIMEOUT = 10

if RETRIES < 1 or RETRIES > 10:
    log.warning("RETRIES %s out of range (1-10). Defaulting to 3 retries.", RETRIES)
    RETRIES = 3

# dynamodb handle
ddb = boto3.resource("dynamodb")
table = ddb.Table(table_name)

log.info(
    "Config loaded: table=%s, tickers=%s, region=%s, lang=%s, timeout=%s, retries=%s",
    table_name,
    ",".join(tickers),
    REGION,
    LANG,
    TIMEOUT,
    RETRIES
)

def _canon_url(url: str) -> str:
    """Canonicalize URL by removing tracking query parameters."""
    parsed = urllib.parse.urlsplit(url)

    # if the URL is not valid, return it as is
    if not parsed.scheme or not parsed.netloc:
        return ""
    
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)

    drop_prefixes = ("utm_", "clk", "soc_src", "soc_trk", "spm", "gclid", "fbclid")

    clean_pairs = [
        (k, v) for (k, v) in pairs
        if not any(k.lower().startswith(prefix) for prefix in drop_prefixes)
    ]

    netloc = parsed.netloc.lower()
    if (parsed.scheme == "http" and netloc.endswith(":80")) or \
       (parsed.scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]

    clean_pairs.sort()

    new_query = urllib.parse.urlencode(clean_pairs, doseq=True)

    clean_url = urllib.parse.urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, new_query, "")
    )
    

    return clean_url

def _sha1(s: str) -> str:
    if not s:
        return ""
    
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _get_with_retry(url:str, timeout: float, retries: int) -> str:
    """GET a URL with basic exponential back-off for 429 and timeouts."""
    MAX_WAIT = 60  # max wait time in seconds
    headers = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SentimentsStocksBot/0.2;"
        " +https://github.com/rzv09/sentiments-stocks)"
    )
    }

    wait = 1
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")
                
                
                log.warning("Received %s status code. Retrying in %s seconds...", r.status_code, wait)
                time.sleep(wait)
                wait = min(wait*2, MAX_WAIT)  # exponential back-off
                continue
            r.raise_for_status()
            return r.text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries -1:
                raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")
            log.warning("Network error %s ... Retrying in %s seconds...", e.__class__.__name__, wait)
            time.sleep(wait)
            wait = min(wait*2, MAX_WAIT)  # exponential back-off
            continue
    raise RuntimeError("Unreachable code reached in _get_with_retry")

def _iso_utc(struct_time_or_none) -> str:
    """Convert feedparser's published_parsed to ISO 8601 UTC string."""
    if not struct_time_or_none: return datetime.datetime.now(datetime.timezone.utc).isoformat()
    # using * unpacks the tuple into positional arguments
    else: return datetime.datetime(*struct_time_or_none[:6], tzinfo=datetime.timezone.utc).isoformat()

def handler(event, context):
    """Lambda entrypoint: fetch RSS for each ticker, canonicalize, dedupe, store."""
    inserted = 0
    dupes = 0

    # keep a set to avoid double-inserting the same URL if it appears for multiple tickers in one run
    seen_urls = set()

    for ticker in tickers:
        feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region={REGION}&lang={LANG}"

        try:
            rss_text = _get_with_retry(feed_url, timeout=TIMEOUT, retries=RETRIES)
        except Exception as e:
            log.warning("Fetch failed for %s: %s", ticker, str(e))
            continue

        parsed = feedparser.parse(rss_text)

        for entry in parsed.entries:
            link = entry.get("link") or ""
            title = entry.get("title") or ""
            published_iso = _iso_utc(entry.get("published_parsed"))
            
            url = _canon_url(link)
            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            url_hash = _sha1(url)
            if not url_hash:
                continue

            # build the item for DynamoDB

            item = {
                "url_hash": url_hash,
                "url": url,
                "ticker": ticker,
                "headline": title,
                "published_utc": published_iso,
                "source": "yahoo_rss",
                "ingested_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            }

            # write to DynamoDB
            try:
                table.put_item(
                    Item=item,
                    ConditionExpression="attribute_not_exists(url_hash)"
                )
                inserted += 1
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                    dupes += 1
                else:
                    raise
    
    log.info("Ingest complete: inserted=%s dupes=%s tickers=%s", inserted, dupes, ",".join(tickers))
    return {"inserted": inserted, "dupes": dupes, "tickers": tickers}




if __name__=="__main__":
    print(handler({}, None))