"""Merge manually approved news candidates into the IV event candidate file."""
import pandas as pd

from .common import ROOT


def main():
    review = pd.read_csv(ROOT / "data/processed/candidate_high_news_days.csv")
    approved = review.query("final_include_high == 1").copy()
    if approved.empty:
        raise SystemExit("No approved news candidates. Complete the review columns first.")
    required = ["surprise_confirmed", "market_attribution_confirmed", "confounder_review"]
    if approved[required].replace("", pd.NA).isna().any().any():
        raise ValueError("Approved candidates require surprise, market-attribution, and confounder review")
    events = pd.read_csv(ROOT / "data/manual/event_candidates.csv")
    existing = set(pd.to_datetime(events["effective_trading_date"]).dt.date)
    rows = []
    for i, r in approved.iterrows():
        d = pd.Timestamp(r["market_date"])
        if d.date() in existing:
            events.loc[pd.to_datetime(events["effective_trading_date"]).dt.date.eq(d.date()), "include_high"] = 1
            continue
        rows.append({"event_id": f"N{d:%Y%m%d}", "event_date": d.date(), "event_time_et": "",
                     "effective_trading_date": d.date(), "event_description": r.get("top_headlines", ""),
                     "direction": "mixed", "channel": "news_panel", "source_url": "",
                     "surprise_reason": r["surprise_confirmed"], "market_session": "close_to_close",
                     "confounding_news": r["confounder_review"], "include_high": 1,
                     "review_status": "approved_news_panel", "notes": r["review_notes"]})
    if rows:
        events = pd.concat([events, pd.DataFrame(rows)], ignore_index=True)
    events.to_csv(ROOT / "data/manual/event_candidates.csv", index=False)
    print(f"Promoted {len(approved)} approved news-panel dates")


if __name__ == "__main__":
    main()
