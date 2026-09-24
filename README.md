# Iran War Risk: Post-Onset Heteroskedasticity Replication

This project adapts Rigobon and Sack's *The Effects of War Risk on U.S. Financial Markets* to the ongoing 2026 Iran war.

## Research object

The sample begins after combat started on 2026-02-28. It does **not** estimate the probability that war begins. The latent factor is unexpected news about the ongoing war's intensity, expected duration, geographic scope, and economic disruption.

An `H` day is a trading day when the variance of this news is unusually high. It may contain escalation or de-escalation news. An `L` day is a nearby trading day with little Iran-war news and no major confounder. Direction is recorded separately from the H/L classification.

Because 2026-02-28 was a Saturday, any onset surprise affecting U.S. markets is assigned to 2026-03-02. The onset is a candidate, not an automatic inclusion.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.build_event_pairs
python -m src.download_market_data
python -m src.run_analysis
pytest -q
```

For the expanded 15-publication news analysis, run this first:

```bash
python -m src.collect_news
python -m src.process_news
python -m src.score_news_days
# Manually review data/processed/candidate_high_news_days.csv
python -m src.promote_news_candidates
python -m src.build_event_pairs
```

Or use the orchestrator:

```bash
python -m src.run_pipeline
# Review candidate_high_news_days.csv
python -m src.run_pipeline --skip-news-download --after-review
```

The first command intentionally stops at the human-review boundary. The second
command will refuse to proceed if no date has been approved.

### News pipeline

The target panel is stored in `data/manual/news_sources.csv`: Reuters, Bloomberg,
Financial Times, Wall Street Journal, CNBC, Associated Press, BBC, CNN, New York
Times, The Guardian, Al Jazeera, Al-Monitor, Times of Israel, IRNA, and Iran
International. The collector searches each domain independently in seven-day
slices, making coverage gaps and record-cap warnings auditable.

The pipeline produces:

- `data/raw/articles_raw.csv`: normalized article metadata plus lawful manual additions
- `data/processed/articles_deduplicated.csv`: exact-deduplicated, scored articles
- `data/processed/event_clusters.csv`: near-duplicate cross-publication event clusters
- `data/processed/daily_news_scores.csv`: daily volume, breadth, event, and market scores
- `data/processed/candidate_high_news_days.csv`: dates requiring human review
- `outputs/news_collection_failures.csv`: missing slices and record-cap warnings
- `outputs/news_source_coverage.csv`: per-publication date and event coverage audit

GDELT timestamps are discovery timestamps, not guaranteed original publication
timestamps. Exact times for final H-day events must be checked on the publisher's
page. The code never bypasses paywalls or treats unavailable article text as observed.

Candidate days must exceed the configured score percentile, appear across at least
three independent source groups, and have coverage from a financial publication.
They still require manual confirmation of surprise, market attribution, and
confounders before `promote_news_candidates` permits them into the event file.

Before running the first command, review `data/manual/event_candidates.csv`. Set `include_high=1` only for events that satisfy the coding protocol. Add excluded macro dates to `data/manual/excluded_dates.csv`.

## Outputs

- `data/processed/event_pairs.csv`: equal-sized H/L sample
- `data/processed/market_changes.csv`: aligned daily changes
- `outputs/table_2_iv_estimates.csv`: the paper's three estimators
- `outputs/table_3_variance_decomposition.csv`: variance shares
- `outputs/event_sample_audit.csv`: merge and missing-data audit

## Exact estimators

For the normalizing variable `x1` and outcome `xj`, define the second moments over H and L days as `W_H` and `W_L`. The implementation uses second moments, matching the paper's zero-mean setup and its statement that variances are average squared changes.

1. `w1`: `(E_H[x1*xj] - E_L[x1*xj]) / (E_H[x1^2] - E_L[x1^2])`
2. `w2`: `(E_H[xj^2] - E_L[xj^2]) / (E_H[x1*xj] - E_L[x1*xj])`
3. `w3`: no-intercept 2SLS combining `w1` and `w2` as instruments.

The main table reports raw loadings, a 25-bp normalizing-variable move for comparison with the paper, and one-standard-deviation H-day shocks. Do not label a positive or negative 2-year yield move as an escalation without separately validating the sign.

## Free-data substitutions

| Construct | Series used | Limitation |
|---|---|---|
| 2Y Treasury | FRED `DGS2` | Constant-maturity, not the paper's fitted off-the-run par yield |
| 10Y Treasury | FRED `DGS10` | Same limitation |
| 10Y breakeven | FRED `T10YIE` | Direct daily estimate |
| Treasury liquidity proxy | `DGS10 - GS10` | Low-frequency/noisy proxy; not an on-the-run premium; exclude from main claims |
| S&P 500 | Yahoo `^GSPC` | Close-to-close return |
| BBB spread | FRED `BAMLC0A4CBBB` | ICE BofA index option-adjusted spread |
| High-yield spread | FRED `BAMLH0A0HYM2` | ICE BofA index option-adjusted spread |
| Oil | Yahoo `CL=F` | Front-month WTI proxy, not a fixed 12-month contract |
| Gold | Yahoo `GC=F` | Futures proxy |
| Broad dollar | FRED `DTWEXBGS` | Broad goods-and-services dollar index |

For an exact replication, replace the liquidity and oil proxy columns in the processed file with an on/off-the-run premium and a constant-maturity 12-month WTI series.

## Identification checks

- H and L counts must be equal.
- H and L dates must be unique U.S. trading days.
- L-day selection excludes a +/- 1 trading-day buffer around H days.
- Major scheduled macro/Fed dates must be manually excluded.
- Report weak first-stage diagnostics and all three estimates.
- Run sensitivity analyses that remove onset, ceasefire, and Hormuz-specific events in turn.
- Treat causal language cautiously if `W_H - W_L` is not approximately rank one or if non-war volatility also changes between samples.

The starter event file is a research queue, not a finished hand-coded dataset. Source URLs and event times must be checked against the original article before final inclusion.
