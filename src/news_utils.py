import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd


ESCALATION = {
    "strike", "strikes", "attack", "attacks", "missile", "missiles", "drone",
    "retaliation", "blockade", "closed", "closure", "killed", "bombing",
    "escalation", "escalates", "threat", "warship", "mine", "seized"
}
DEESCALATION = {
    "ceasefire", "truce", "talks", "negotiation", "negotiations", "deal",
    "reopen", "reopening", "peace", "halt", "pause", "diplomacy", "agreement"
}
MARKET = {
    "oil", "crude", "brent", "wti", "market", "markets", "stocks", "bonds",
    "treasury", "yield", "dollar", "gold", "inflation", "shipping", "tanker",
    "hormuz", "pipeline", "refinery", "energy", "supply"
}


def normalize_url(url):
    if not isinstance(url, str):
        return ""
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix("www."),
                       p.path.rstrip("/"), "", ""))


def normalize_text(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return " ".join(text.split())


def stable_id(*parts):
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def keyword_scores(text):
    words = set(normalize_text(text).split())
    e, d, m = len(words & ESCALATION), len(words & DEESCALATION), len(words & MARKET)
    if e and d:
        direction = "mixed"
    elif e:
        direction = "escalation"
    elif d:
        direction = "deescalation"
    else:
        direction = "unclear"
    return direction, min(e / 3, 1.0), min(d / 3, 1.0), min(m / 3, 1.0)


def assign_market_date(timestamp_utc, close_hour=16):
    ts = pd.Timestamp(timestamp_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert("America/New_York")
    date = et.normalize().tz_localize(None)
    if et.hour >= close_hour:
        date += pd.Timedelta(days=1)
    while date.weekday() >= 5:
        date += pd.Timedelta(days=1)
    return et, date


def robust_trailing_z(series, window=14):
    values = pd.Series(series, dtype=float)
    med = values.shift(1).rolling(window, min_periods=max(5, window // 2)).median()
    mad = (values.shift(1) - med).abs().rolling(window, min_periods=max(5, window // 2)).median()
    scale = 1.4826 * mad
    z = (values - med) / scale.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)
