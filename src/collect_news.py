"""Collect historical GDELT news metadata through Google BigQuery.

Queries the public ``gdelt-bq.gdeltv2.gkg_partitioned`` table in monthly
partition windows. The caller's Google Cloud project supplies BigQuery billing
and quota; no GDELT API key is required. Each completed month is checkpointed.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from .common import ROOT, ensure_dirs, load_config
from .news_utils import assign_market_date, normalize_url, stable_id

TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
CHECKPOINT_DIR = ROOT / "data/raw/gdelt_bigquery"

IRAN_PATTERN = r"(^|[^a-z])(iran|iranian|tehran|irgc)([^a-z]|$)"
TOPIC_PATTERN = (
    r"(^|[^a-z])(war|strike|strikes|missile|missiles|drone|drones|attack|attacks|"
    r"retaliation|ceasefire|truce|negotiation|negotiations|talks|blockade|hormuz|"
    r"tanker|tankers|shipping|oil|nuclear|military|airspace)([^a-z]|$)"
)

QUERY = f"""
WITH filtered AS (
  SELECT
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(`DATE` AS STRING)) AS published_at_utc,
    LOWER(NET.REG_DOMAIN(DocumentIdentifier)) AS domain,
    SourceCommonName AS source_common_name,
    DocumentIdentifier AS url,
    REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>([^<]*)</PAGE_TITLE>') AS headline,
    REGEXP_EXTRACT(Extras, r'<PAGE_DESCRIPTION>([^<]*)</PAGE_DESCRIPTION>') AS description,
    V2Themes AS themes,
    V2Tone AS tone,
    SharingImage AS sharing_image
  FROM `{TABLE}`
  WHERE _PARTITIONTIME >= TIMESTAMP(@start_date)
    AND _PARTITIONTIME < TIMESTAMP(@end_date)
    AND LOWER(NET.REG_DOMAIN(DocumentIdentifier)) IN UNNEST(@domains)
    AND REGEXP_CONTAINS(
      LOWER(CONCAT(
        COALESCE(Extras, ''), ' ', COALESCE(V2Themes, ''), ' ',
        COALESCE(V2Locations, ''), ' ', COALESCE(V2Organizations, '')
      )), @iran_pattern
    )
    AND REGEXP_CONTAINS(
      LOWER(CONCAT(COALESCE(Extras, ''), ' ', COALESCE(V2Themes, ''))),
      @topic_pattern
    )
)
SELECT *
FROM filtered
QUALIFY ROW_NUMBER() OVER (PARTITION BY url ORDER BY published_at_utc) = 1
ORDER BY published_at_utc
"""


def authenticate_colab():
    try:
        from google.colab import auth
    except ImportError as exc:
        raise RuntimeError("--authenticate-colab is only available in Google Colab") from exc
    auth.authenticate_user()


def month_windows(start, end):
    left = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    while left < end:
        next_month = left.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
        right = min(next_month, end)
        yield left, right
        left = right


def query_config(start, end, domains, dry_run=False, maximum_bytes_billed=None):
    config = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=not dry_run,
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start.date()),
            bigquery.ScalarQueryParameter("end_date", "DATE", end.date()),
            bigquery.ArrayQueryParameter("domains", "STRING", domains),
            bigquery.ScalarQueryParameter("iran_pattern", "STRING", IRAN_PATTERN),
            bigquery.ScalarQueryParameter("topic_pattern", "STRING", TOPIC_PATTERN),
        ],
    )
    if maximum_bytes_billed:
        config.maximum_bytes_billed = int(maximum_bytes_billed)
    return config


def source_lookup(sources):
    return sources.set_index(sources["domain"].str.lower()).to_dict("index")


def standardize_bigquery(frame, sources, close_hour):
    if frame.empty:
        return pd.DataFrame(columns=[
            "article_id", "publication", "domain", "source_group",
            "financial_source", "perspective", "published_at_utc",
            "published_at_et", "market_date", "headline", "url", "language",
            "source_country", "article_text", "collection_method",
            "manual_review_status", "description", "themes", "tone", "sharing_image",
        ])
    lookup = source_lookup(sources)
    rows = []
    for row in frame.itertuples(index=False):
        domain = str(row.domain).lower()
        source = lookup.get(domain)
        if source is None:
            continue
        timestamp = pd.Timestamp(row.published_at_utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        et, market_date = assign_market_date(timestamp, close_hour)
        url = normalize_url(row.url)
        headline = row.headline if isinstance(row.headline, str) else ""
        description = row.description if isinstance(row.description, str) else ""
        rows.append({
            "article_id": stable_id(url, headline, timestamp.isoformat()),
            "publication": source["publication"], "domain": domain,
            "source_group": source["source_group"],
            "financial_source": int(source["financial_source"]),
            "perspective": source["perspective"],
            "published_at_utc": timestamp.isoformat(),
            "published_at_et": et.isoformat(),
            "market_date": market_date.date().isoformat(),
            "headline": headline or description, "url": url,
            "language": "English", "source_country": "",
            "article_text": description, "collection_method": "gdelt_gkg_bigquery",
            "manual_review_status": "unreviewed", "description": description,
            "themes": row.themes, "tone": row.tone, "sharing_image": row.sharing_image,
        })
    return pd.DataFrame(rows)


def add_manual_articles(frame, sources, close_hour):
    manual = pd.read_csv(ROOT / "data/manual/articles_manual.csv")
    if manual.empty:
        return frame
    lookup, extras = source_lookup(sources), []
    publication_to_source = {v["publication"]: (k, v) for k, v in lookup.items()}
    for row in manual.itertuples(index=False):
        if row.publication not in publication_to_source:
            raise ValueError(f"Unknown manual publication: {row.publication}")
        domain, source = publication_to_source[row.publication]
        timestamp = pd.to_datetime(row.published_at_utc, utc=True)
        et, market_date = assign_market_date(timestamp, close_hour)
        url = normalize_url(row.url)
        extras.append({
            "article_id": stable_id(url, row.headline, timestamp.isoformat()),
            "publication": row.publication, "domain": domain,
            "source_group": source["source_group"],
            "financial_source": int(source["financial_source"]),
            "perspective": source["perspective"],
            "published_at_utc": timestamp.isoformat(), "published_at_et": et.isoformat(),
            "market_date": market_date.date().isoformat(), "headline": row.headline,
            "url": url, "language": "English", "source_country": "",
            "article_text": getattr(row, "article_text", ""),
            "collection_method": "manual", "manual_review_status": "unreviewed",
            "description": "", "themes": "", "tone": "", "sharing_image": "",
        })
    return pd.concat([frame, pd.DataFrame(extras)], ignore_index=True).drop_duplicates("article_id")


def format_bytes(value):
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud project used for BigQuery billing/quota")
    parser.add_argument("--location", default="US")
    parser.add_argument("--authenticate-colab", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate bytes only; do not download rows")
    parser.add_argument("--fresh", action="store_true",
                        help="Rerun and overwrite monthly checkpoints")
    parser.add_argument("--maximum-bytes-billed", type=int, default=None,
                        help="Optional per-query safety cap in bytes")
    args = parser.parse_args(argv)
    if not args.project:
        raise SystemExit("Provide --project YOUR_GCP_PROJECT_ID or set GOOGLE_CLOUD_PROJECT.")
    if args.authenticate_colab:
        authenticate_colab()

    ensure_dirs()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()["news"]
    sources = pd.read_csv(ROOT / "data/manual/news_sources.csv").query("active == 1")
    domains = sorted(sources["domain"].str.lower().unique().tolist())
    windows = list(month_windows(cfg["start"], cfg["end"]))
    client = bigquery.Client(project=args.project, location=args.location)

    if args.dry_run:
        total = 0
        for index, (start, end) in enumerate(windows, 1):
            job = client.query(QUERY, job_config=query_config(
                start, end, domains, dry_run=True,
                maximum_bytes_billed=args.maximum_bytes_billed))
            total += job.total_bytes_processed
            print(f"[{index}/{len(windows)}] {start.date()}..{end.date()}: "
                  f"{format_bytes(job.total_bytes_processed)}", flush=True)
        print(f"Estimated total scanned: {format_bytes(total)}", flush=True)
        return 0

    started, monthly_files = time.monotonic(), []
    for index, (start, end) in enumerate(windows, 1):
        path = CHECKPOINT_DIR / f"articles_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if path.exists() and not args.fresh:
            month = pd.read_csv(path)
            status = "checkpoint"
        else:
            job = client.query(QUERY, job_config=query_config(
                start, end, domains, maximum_bytes_billed=args.maximum_bytes_billed))
            raw = job.result().to_dataframe(create_bqstorage_client=False)
            month = standardize_bigquery(raw, sources, int(cfg["close_hour_et"]))
            month.to_csv(path, index=False)
            status = f"queried {format_bytes(job.total_bytes_processed)}"
        monthly_files.append(path)
        elapsed = time.monotonic() - started
        eta = elapsed / index * (len(windows) - index)
        print(f"[{index}/{len(windows)}] {100*index/len(windows):5.1f}% | "
              f"{start:%Y-%m-%d}..{end:%Y-%m-%d} | {len(month):,} articles | "
              f"{status} | ETA {eta/60:.1f} min", flush=True)

    frames = [pd.read_csv(path) for path in monthly_files]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame = frame.drop_duplicates("article_id").sort_values("published_at_utc")
    frame = add_manual_articles(frame, sources, int(cfg["close_hour_et"]))
    frame.to_csv(ROOT / "data/raw/articles_raw.csv", index=False)
    coverage = frame.groupby("publication").size().rename("article_count").reset_index()
    coverage.to_csv(ROOT / "outputs/news_collection_bigquery_coverage.csv", index=False)
    print(f"Finished: {len(frame):,} unique articles from "
          f"{frame['publication'].nunique() if not frame.empty else 0} publications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
