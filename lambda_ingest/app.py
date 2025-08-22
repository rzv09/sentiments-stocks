import os
import time
import hashlib
import logging
import datetime
import urllib.parse
import requests
import feedparser
import boto3

# logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger(__name__)

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