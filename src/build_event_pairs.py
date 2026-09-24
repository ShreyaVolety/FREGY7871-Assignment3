"""Construct reproducible, equal-sized high- and low-news samples."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from scipy.optimize import linear_sum_assignment

from .common import ROOT, ensure_dirs, load_config


def trading_grid(start, end):
    # Start from federal holidays, add Good Friday, and remove federal holidays
    # on which the NYSE normally remains open.
    calendar = USFederalHolidayCalendar()
    rules = [
        rule for rule in calendar.rules
        if rule.name not in {"Columbus Day", "Veterans Day"}
    ] + [GoodFriday]
    calendar.rules = rules
    cbd = CustomBusinessDay(calendar=calendar)
    return pd.DatetimeIndex(pd.date_range(start, end, freq=cbd))


def choose_low_days(high_dates, grid, excluded, daily_scores, low_quantile=0.50):
    """Assign each H day a unique nearby date from the low-news candidate pool."""
    high = set(pd.to_datetime(high_dates).dt.normalize())
    banned = set(pd.to_datetime(excluded)) | high
    positions = {d: i for i, d in enumerate(grid)}
    for h in high:
        if h not in positions:
            raise ValueError(f"H date {h.date()} is not in the trading-day grid")

    scores = daily_scores.copy()
    scores.index = pd.to_datetime(scores.index).normalize()
    pool = scores.loc[scores.index.intersection(pd.DatetimeIndex(set(grid) - banned))].copy()
    pool = pool.loc[~pool["news_intensity_score"].isna()]
    cutoff = pool["news_intensity_score"].quantile(float(low_quantile))
    pool = pool.loc[pool["news_intensity_score"] <= cutoff].sort_index()
    if len(pool) < len(high):
        raise ValueError(
            f"Need {len(high)} L days but the bottom {low_quantile:.0%} contains "
            f"only {len(pool)} eligible dates; increase sample.low_news_percentile."
        )

    high_sorted = pd.DatetimeIndex(sorted(high))
    low_dates = pd.DatetimeIndex(pool.index)
    high_pos = np.array([positions[d] for d in high_sorted])[:, None]
    low_pos = np.array([positions[d] for d in low_dates])[None, :]

    # Minimize total trading-day distance across all unique assignments. A tiny
    # intensity-rank term breaks equal-distance ties in favor of quieter dates.
    distances = np.abs(high_pos - low_pos).astype(float)
    intensity_rank = pool["news_intensity_score"].rank(pct=True).to_numpy()[None, :]
    row_index, col_index = linear_sum_assignment(distances + 0.001 * intensity_rank)
    return [(high_sorted[i], low_dates[j]) for i, j in zip(row_index, col_index)]


def main():
    ensure_dirs()
    cfg = load_config()["sample"]
    events = pd.read_csv(ROOT / "data/manual/event_candidates.csv")
    selected = events.loc[events["include_high"].astype(str).eq("1")].copy()
    if selected.empty:
        raise SystemExit("No H days selected. Review event_candidates.csv and set include_high=1.")
    selected["effective_trading_date"] = pd.to_datetime(selected["effective_trading_date"])
    if selected["effective_trading_date"].duplicated().any():
        dup = selected.loc[selected["effective_trading_date"].duplicated(False), "effective_trading_date"]
        raise ValueError(f"Multiple selected events map to the same H day: {dup.dt.date.tolist()}")
    exclusions = pd.read_csv(ROOT / "data/manual/excluded_dates.csv")
    daily_scores = pd.read_csv(
        ROOT / "data/processed/daily_news_scores.csv",
        parse_dates=["market_date"],
    ).set_index("market_date")
    grid = trading_grid(cfg["start"], cfg["end"])
    pairs = choose_low_days(
        selected["effective_trading_date"], grid, exclusions["date"], daily_scores,
        float(cfg.get("low_news_percentile", 0.50)),
    )
    rows = []
    by_date = selected.set_index("effective_trading_date")
    for pair_id, (h, l) in enumerate(pairs, 1):
        event = by_date.loc[h]
        common = {"pair_id": pair_id, "event_id": event["event_id"],
                  "direction": event["direction"], "event_description": event["event_description"]}
        rows.extend([{**common, "sample": "H", "date": h.date()},
                     {**common, "sample": "L", "date": l.date()}])
    out = pd.DataFrame(rows).sort_values(["pair_id", "sample"])
    if out.groupby("sample").size().nunique() != 1 or out["date"].duplicated().any():
        raise AssertionError("H/L balance or date uniqueness failed")
    out.to_csv(ROOT / "data/processed/event_pairs.csv", index=False)
    l_scores = daily_scores.reindex(pd.DatetimeIndex([p[1] for p in pairs]))["news_intensity_score"]
    positions = {d: i for i, d in enumerate(grid)}
    distances = [abs(positions[h] - positions[l]) for h, l in pairs]
    print(f"Wrote {len(pairs)} H and {len(pairs)} L observations")
    print(
        "L-day news intensity: "
        f"median={l_scores.median():.3f}, max={l_scores.max():.3f}. "
        f"Trading-day distance: median={np.median(distances):.1f}, "
        f"max={max(distances)}."
    )


if __name__ == "__main__":
    main()
