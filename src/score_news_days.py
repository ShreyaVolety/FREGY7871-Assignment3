"""Aggregate article/event evidence into reviewable candidate H days."""
import numpy as np
import pandas as pd

from .common import ROOT, ensure_dirs, load_config
from .news_utils import robust_trailing_z


def global_robust_z(series: pd.Series) -> pd.Series:
    """Score every post-onset day against the same full-period robust baseline."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad

    # Some breadth measures can have MAD=0. Standard deviation is a stable
    # fallback; a constant series contains no intensity information.
    if not np.isfinite(scale) or scale <= 0:
        scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(0.0, index=values.index)
    return ((values - median) / scale).clip(-10, 10)


def main():
    ensure_dirs()
    cfg = load_config()["news"]
    articles = pd.read_csv(ROOT / "data/processed/articles_deduplicated.csv",
                           parse_dates=["market_date", "published_at_utc"])
    coverage = articles.groupby(["publication", "source_group"]).agg(
        first_seen=("published_at_utc", "min"), last_seen=("published_at_utc", "max"),
        article_count=("article_id", "size"),
        covered_market_days=("market_date", "nunique"),
        distinct_events=("event_cluster_id", "nunique"),
    ).reset_index()
    coverage.to_csv(ROOT / "outputs/news_source_coverage.csv", index=False)
    daily = articles.groupby("market_date").agg(
        article_volume=("article_id", "size"),
        publication_breadth=("publication", "nunique"),
        source_group_breadth=("source_group", "nunique"),
        event_breadth=("event_cluster_id", "nunique"),
        financial_publication_breadth=("financial_source", "sum"),
        market_relevance=("market_relevance_score", "max"),
        escalation_signal=("escalation_score", "max"),
        deescalation_signal=("deescalation_score", "max"),
    ).sort_index()
    grid = pd.date_range(cfg["start"], cfg["end"], freq="B")
    daily = daily.reindex(grid, fill_value=0)
    window = int(cfg["trailing_baseline_days"])
    for col in ["article_volume", "publication_breadth", "source_group_breadth", "event_breadth"]:
        # Global scores drive candidate selection, so March dates are directly
        # comparable with every later date in the post-onset sample.
        daily[f"z_{col}"] = global_robust_z(daily[col])
        # Retain backward-looking scores for diagnostics, but do not use them
        # to decide whether an early date can be a candidate.
        daily[f"trailing_z_{col}"] = robust_trailing_z(daily[col], window)
    daily["news_intensity_score"] = (
        .20*daily["z_article_volume"] + .20*daily["z_publication_breadth"] +
        .25*daily["z_source_group_breadth"] + .20*daily["z_event_breadth"] +
        .15*daily["market_relevance"]
    )
    cutoff = daily["news_intensity_score"].quantile(float(cfg["candidate_percentile"]))
    daily["candidate_high"] = (
        (daily["news_intensity_score"] >= cutoff) &
        (daily["source_group_breadth"] >= int(cfg["minimum_source_groups"])) &
        (daily["financial_publication_breadth"] >= 1)
    ).astype(int)
    daily.index.name = "market_date"
    daily.to_csv(ROOT / "data/processed/daily_news_scores.csv")
    candidates = daily.query("candidate_high == 1").reset_index()
    top = (articles.merge(candidates[["market_date"]], on="market_date")
           .sort_values(["market_date", "market_relevance_score", "published_at_utc"],
                        ascending=[True, False, True])
           .groupby("market_date").head(8)
           .groupby("market_date")["headline"].apply(" | ".join).rename("top_headlines"))
    candidates = candidates.merge(top, on="market_date", how="left")
    candidates["surprise_confirmed"] = ""
    candidates["market_attribution_confirmed"] = ""
    candidates["confounder_review"] = ""
    candidates["final_include_high"] = 0
    candidates["review_notes"] = ""
    candidates.to_csv(ROOT / "data/processed/candidate_high_news_days.csv", index=False)
    print(
        f"Generated {len(candidates)} candidate H days at cutoff {cutoff:.3f} "
        f"(percentile={float(cfg['candidate_percentile']):.2f}; "
        "global post-onset robust normalization)"
    )


if __name__ == "__main__":
    main()
