import pandas as pd
from src.news_utils import assign_market_date, keyword_scores, normalize_url


def test_after_close_rolls_forward():
    et, market_date = assign_market_date("2026-07-10T21:30:00Z")
    assert et.hour == 17
    assert market_date == pd.Timestamp("2026-07-13")


def test_direction_and_market_keywords():
    direction, esc, deesc, market = keyword_scores("Ceasefire talks reopen Hormuz oil shipping")
    assert direction == "deescalation" and deesc > 0 and market > 0 and esc == 0


def test_url_normalization():
    assert normalize_url("https://www.example.com/a/?utm_source=x") == "https://example.com/a"
