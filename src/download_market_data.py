"""Download free daily proxies and construct changes in the paper's units."""
from io import StringIO
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from .common import ROOT, ensure_dirs, load_config


def fred_series(series_id, name, start, end):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = ["date", name]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.set_index("date")


def yahoo_series(ticker, name, start, end):
    raw = yf.download(ticker, start=start, end=(pd.Timestamp(end)+pd.Timedelta(days=1)).date(),
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No Yahoo data returned for {ticker}")
    col = raw["Adj Close"] if "Adj Close" in raw else raw["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.rename(name).to_frame()


def main():
    ensure_dirs()
    cfg = load_config()
    start = (pd.Timestamp(cfg["sample"]["start"]) - pd.Timedelta(days=40)).date().isoformat()
    end = cfg["sample"]["end"]
    levels = []
    for name, sid in cfg["data"]["fred"].items():
        levels.append(fred_series(sid, name, start, end))
    for name, ticker in cfg["data"]["yahoo"].items():
        levels.append(yahoo_series(ticker, name, start, end))
    data = pd.concat(levels, axis=1).sort_index()
    # Monthly GS10 is forward-filled only to create a documented rough proxy.
    data["gs10_monthly"] = data["gs10_monthly"].ffill()
    data["liquidity_level_proxy"] = data["dgs10"] - data["gs10_monthly"]
    out = pd.DataFrame(index=data.index)
    for col in ["dgs2", "dgs10", "breakeven10", "bbb_spread", "hy_spread"]:
        out[col] = data[col].diff()
    out["liquidity_proxy"] = data["liquidity_level_proxy"].diff()
    for col in ["sp500", "broad_dollar"]:
        out[col] = 100 * np.log(data[col] / data[col].shift(1))
    for col in ["oil_front", "gold"]:
        out[col] = data[col].diff()
    data.to_csv(ROOT / "data/raw/market_levels.csv", index_label="date")
    out.loc[cfg["sample"]["start"]:end].to_csv(
        ROOT / "data/processed/market_changes.csv", index_label="date")
    print(f"Wrote {len(out.loc[cfg['sample']['start']:end])} daily observations")


if __name__ == "__main__":
    main()
