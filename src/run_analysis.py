import numpy as np
import pandas as pd

from .common import ROOT, ensure_dirs, load_config
from .estimators import estimate_pair


def main():
    ensure_dirs()
    cfg = load_config()
    pairs = pd.read_csv(ROOT / "data/processed/event_pairs.csv", parse_dates=["date"])
    market = pd.read_csv(ROOT / "data/processed/market_changes.csv", parse_dates=["date"])
    merged = pairs.merge(market, on="date", how="left", validate="one_to_one")
    merged.to_csv(ROOT / "outputs/event_sample_audit.csv", index=False)
    variables = list(cfg["units"])
    missing = merged[variables].isna().sum()
    if missing.any():
        raise ValueError(f"Missing event-day data:\n{missing[missing.gt(0)]}")
    h = merged.query("sample == 'H'").sort_values("pair_id")
    l = merged.query("sample == 'L'").sort_values("pair_id")
    x1_name = cfg["normalization"]["variable"]
    move = float(cfg["normalization"]["comparison_move"])
    rows, variance_rows = [], []
    x1_h, x1_l = h[x1_name].to_numpy(), l[x1_name].to_numpy()
    for name in variables:
        if name == x1_name:
            continue
        est = estimate_pair(x1_h, x1_l, h[name].to_numpy(), l[name].to_numpy())
        sd_shock = np.sqrt(max(est["delta_var_x1"], 0.0))
        rows.append({"variable": name, "units": cfg["units"][name], **est,
                     "w1_effect_comparison_move": est["w1_beta"]*move,
                     "w2_effect_comparison_move": est["w2_beta"]*move,
                     "w3_effect_comparison_move": est["w3_beta"]*move,
                     "w3_effect_one_sd": est["w3_beta"]*sd_shock})
        predicted = est["w3_beta"]**2 * est["delta_var_x1"]
        total_period_var = (len(h)*est["var_h_xj"] + len(l)*est["var_l_xj"]) / (len(h)+len(l))
        variance_rows.append({"variable": name, "var_l": est["var_l_xj"],
                              "var_h": est["var_h_xj"], "predicted_change": predicted,
                              "pct_explained_h_lower_bound": 100*predicted/est["var_h_xj"] if est["var_h_xj"] else np.nan,
                              "pct_explained_paired_period": 100*(len(h)*predicted)/((len(h)+len(l))*total_period_var) if total_period_var else np.nan})
    pd.DataFrame(rows).to_csv(ROOT / "outputs/table_2_iv_estimates.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(ROOT / "outputs/table_3_variance_decomposition.csv", index=False)
    print(f"Estimated {len(rows)} outcomes using {len(h)} H/L pairs")


if __name__ == "__main__":
    main()
