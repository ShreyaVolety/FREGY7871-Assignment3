# Event coding protocol

## Include as H only when all are true

1. The information concerns the post-onset Iran war.
2. It was not already substantially anticipated.
3. It plausibly changed expected intensity, duration, geographic scope, or economic disruption.
4. Contemporaneous financial commentary identifies it as an important driver of at least one study market.
5. The effective date and U.S. market session are known.
6. No unrelated shock dominates the day, or the event is retained only in a clearly labeled robustness sample.

## Direction

Use `escalation`, `deescalation`, or `mixed`. Direction never determines H status. Do not infer direction from the sign of the 2-year yield.

## Effective trading date

- Before 4:00 p.m. ET on a trading day: same date.
- After 4:00 p.m. ET, weekend, or U.S. market holiday: next trading day.
- For news that accumulated overnight, record the earliest timestamp and explain the assignment.

## Low-day rules

The script chooses the nearest unused trading day within the configured window, after removing H dates, a one-trading-day buffer around H, and manually excluded dates. Review every proposed L date for CPI, payrolls, PCE, GDP, FOMC, major Treasury announcements, tariff/fiscal shocks, other geopolitical news, and exceptional firm-specific index moves.

## Required archive fields

Keep the headline, publisher, exact timestamp, stable URL, short surprise justification, market-attribution quotation or paraphrase, release session, confounder assessment, coder, and review status. A chronology article may identify candidates, but final coding should cite contemporaneous reporting.
