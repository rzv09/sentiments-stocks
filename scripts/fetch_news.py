#!/usr/bin/env python3
"""
fetch_news.py — Sentiments‑Stocks project
----------------------------------------
Pulls the latest finance headlines from Yahoo Finance’s public RSS feed
(for one or more ticker symbols) and stores them in the **raw_news** table
of the Postgres database configured via the `DB_URL` environment variable.

Additions in v0.2
-----------------
* Graceful handling of HTTP 429 (rate‑limit) and transient network errors
  with exponential back‑off retries.
* Custom User‑Agent header so Yahoo doesn’t treat the script as a bot.
* Optional `--retries` CLI flag (default 3).

Usage (host or in Docker‑exec):
    python scripts/fetch_news.py --tickers AAPL,MSFT --limit 50

Environment variables:
    DB_URL   SQLAlchemy‑compatible URL, e.g.
             postgresql+psycopg://sentstocks:sentstocks@localhost:5432/sentstocks
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import uuid
from typing import Iterable, List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup, FeatureNotFound
from requests.exceptions import HTTPError, Timeout
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine, text

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SentimentsStocksBot/0.2;"
        " +https://github.com/rzv09/sentiments-stocks)"
    )
}


def _ensure_raw_news_table(engine) -> Table:
    """Create the *raw_news* table if it doesn’t exist; return the Table obj."""
    metadata = MetaData()
    raw_news = Table(
        "raw_news",
        metadata,
        Column("news_id", String(36), primary_key=True),
        Column("source", String(50)),
        Column("headline", String),
        Column("url", String),
        Column("tickers_raw", String),
        Column("published_utc", DateTime),
        Column("ingested_at_utc", DateTime, default=dt.datetime.utcnow),
        extend_existing=True,
    )
    metadata.create_all(engine)  # no‑op if table already exists
    return raw_news


def _get_with_retry(url: str, headers: dict, retries: int = 3, backoff: int = 5):
    """GET a URL with basic exponential back‑off for 429 and timeouts."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15, headers=headers)
            # Explicitly treat 429 as retryable
            if resp.status_code == 429:
                raise HTTPError("429 Too Many Requests", response=resp)
            resp.raise_for_status()
            return resp
        except (HTTPError, Timeout) as err:
            if isinstance(err, HTTPError) and err.response.status_code != 429:
                # Non‑retryable HTTP error (e.g., 404). Reraise immediately.
                raise
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)  # exponential back‑off
            print(f"Warning: {err}. Retrying in {wait}s ...", file=sys.stderr)
            time.sleep(wait)
    # Should never reach here
    raise RuntimeError("_get_with_retry logic fell through")


def _fetch_yahoo_rss(tickers: List[str], retries: int = 3) -> Iterable[dict]:
    """Yield dicts for each <item> in Yahoo Finance combined‑ticker RSS feed."""
    tickers_str = ",".join(tickers)
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?"
        f"s={quote_plus(tickers_str)}&region=US&lang=en-US"
    )
    resp = _get_with_retry(url, headers=DEFAULT_HEADERS, retries=retries)

    try:
        soup = BeautifulSoup(resp.content, "lxml-xml")  # prefer lxml if available
    except FeatureNotFound:
        soup = BeautifulSoup(resp.content, "xml")  # fallback to built‑in
    for item in soup.find_all("item"):
        yield {
            "source": "YahooFinanceRSS",
            "headline": item.title.get_text(strip=True),
            "url": item.link.get_text(strip=True),
            "tickers_raw": tickers_str,
            "published_utc": dt.datetime.strptime(
                item.pubDate.get_text(strip=True), "%a, %d %b %Y %H:%M:%S %z"
            )
            .astimezone(dt.timezone.utc)
            .replace(tzinfo=None),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest headlines into Postgres.")
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT",
        help="Comma‑separated ticker symbols (max 10).",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum headlines to ingest."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries on 429/timeout errors."
    )
    args = parser.parse_args(argv)

    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    db_url = os.getenv(
        "DB_URL", "postgresql+psycopg://sentstocks:sentstocks@localhost:5432/sentstocks"
    )
    engine = create_engine(db_url, future=True, echo=False)

    table = _ensure_raw_news_table(engine)

    try:
        headlines = list(_fetch_yahoo_rss(tickers, retries=args.retries))[: args.limit]
    except Exception as err:
        print(f"ERROR: Failed to fetch RSS: {err}", file=sys.stderr)
        sys.exit(2)

    if not headlines:
        print("No headlines fetched — RSS feed may be empty.", file=sys.stderr)
        sys.exit(1)

    inserted = 0
    with engine.begin() as conn:
        for row in headlines:
            # Deduplicate by URL
            if conn.execute(text("SELECT 1 FROM raw_news WHERE url = :u"), {"u": row["url"]}).first():
                continue
            row["news_id"] = str(uuid.uuid4())
            row["ingested_at_utc"] = dt.datetime.utcnow()
            conn.execute(table.insert().values(**row))
            inserted += 1

    print(f"Ingested {inserted} new headline(s) out of {len(headlines)} fetched.")


if __name__ == "__main__":
    main()
