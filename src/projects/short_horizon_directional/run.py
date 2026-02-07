from __future__ import annotations

import gc
import pathlib

import numpy as np
import pandas as pd

from src.loaders.parquets_to_df import parquet_to_df
from src.bars.create_time_bars_w_bbo import create_time_bars_w_bbo
from src.features.med_mad_tod_norm import add_med_mad_norm
from src.features.percentile_burst import add_rolling_percentile_burst
from src.labels.tail_label import build_tail_labels_session_aware
from src.projects.vol_state_detection.run import main as run_vol_states
from src.models.bbo_xgboost_baseline import fit_baseline_bbo_xgb

# =========================
# CONFIG
# =========================
TRADES_PATH = pathlib.Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\trades")
BBO_PATH    = pathlib.Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\mbp1")

DATE_SLICE = ("2025-01-01", "2025-12-20")
TIME_INT   = "2s"
BOOL_FILL  = True

HORIZON        = 5
SPREAD_MULT    = 1.5
TRAIN_FRAC     = 0.70

# Selection quantile for PnL proxy (top-probability tail)
Q_PROBA_THRESH = 0.95

# =========================
# PIPELINE
# =========================
def load_bars() -> pd.DataFrame:
    trades = parquet_to_df(TRADES_PATH).loc[DATE_SLICE[0] : DATE_SLICE[1]]
    bbo    = parquet_to_df(BBO_PATH).loc[DATE_SLICE[0] : DATE_SLICE[1]]

    bars = create_time_bars_w_bbo(trades, bbo, time_int=TIME_INT, bool_fill=BOOL_FILL)

    del trades, bbo
    gc.collect()

    return bars


def add_hl_state_dummies(bars: pd.DataFrame) -> pd.DataFrame:
    vol_states = run_vol_states(return_df=True)

    bars = pd.merge_asof(
        bars.sort_index(),
        vol_states.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    del vol_states
    gc.collect()

    # one-hot HL states only (Int8)
    for i in range(3):
        bars[f"states_hl_{i}"] = bars["states_hl"].eq(i).astype("Int8")

    # drop raw state col(s)
    bars = bars.drop(columns=["states_hl"])
    bars = bars.drop(columns=["states_rv"])

    return bars


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars["log_volume"] = np.log1p(bars["volume"])

    tmp = add_med_mad_norm(bars, feature_col="log_volume")
    bars["log_volume_mad_z"] = tmp["log_volume_mad_z"]

    tmp = add_rolling_percentile_burst(bars, feature_col="log_volume")
    bars["log_volume_pct_score"] = tmp["log_volume_pct_score"]

    bars["log_bbo_updates"] = np.log1p(bars["bbo_updates"])
    bars["mp_over_spread"]  = bars["microprice_diff"] / (bars["spread"] + 1e-9)
    bars["net_aggr"]        = bars["aggr_buy_vol"] - bars["aggr_sell_vol"]

    bars = bars.dropna(subset=["log_volume_mad_z", "log_volume_pct_score"])

    return bars


def apply_hl_gate(
        out: pd.DataFrame,
        bars: pd.DataFrame,
        gate_state: int,
) -> pd.DataFrame:
    
    col = f"states_hl_{gate_state}"
    if col not in bars.columns:
        raise KeyError(f"Missing HL gate column: {col}")

    gate = bars[col].rename(col)
    out2 = out.join(gate, how="inner")
    out2 = out2.loc[out2[col] == 1].drop(columns=[col])

    return out2


# =========================
# RUN
# =========================
def main():

    bars = load_bars()

    bars = add_hl_state_dummies(bars)

    bars = add_features(bars)

    bars = build_tail_labels_session_aware(bars, horizon=HORIZON, spread_mult=SPREAD_MULT)
    bars = bars.dropna(subset=["y_up", "y_down"])

    feature_cols = [
        "imbalance",
        "spread",
        "bbo_updates",
        "mp_over_spread",
        "net_aggr",
        "log_volume_mad_z",
    ]

    model, metrics, p_test = fit_baseline_bbo_xgb(
        bars,
        label_col="y_up",
        feature_cols=feature_cols,
        train_frac=TRAIN_FRAC,
    )

    out = pd.DataFrame({"p_y_up": p_test}).join(bars["mid"], how="inner")
    out["fwd_dmid"] = out["mid"].shift(-HORIZON) - out["mid"]
    out = out.dropna(subset=["fwd_dmid"])

    def summarize_proxy(df: pd.DataFrame, prob_col: str = "p_y_up") -> dict:

        thresh = df[prob_col].quantile(Q_PROBA_THRESH)
        sel = df.loc[df[prob_col] >= thresh]
        return {
            "rows": int(len(df)),
            "q": float(Q_PROBA_THRESH),
            "thresh": float(thresh),
            "n_top": int(len(sel)),
            "mean_fwd_dmid": float(sel["fwd_dmid"].mean()),
            "median_fwd_dmid": float(sel["fwd_dmid"].median()),
            "hit_rate": float((sel["fwd_dmid"] > 0).mean()),
        }

    proxy_all = summarize_proxy(out)

    print("=== MODEL METRICS (TEST) ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v}")

    print("\n=== PNL PROXY (NO GATE) ===")
    for k, v in proxy_all.items():
        print(f"{k:>16}: {v}")

    print("\n=== PNL PROXY (HL GATED) ===")
    out_gated_dict: dict[int, pd.DataFrame] = {}

    for s in range(3):
        out_g = apply_hl_gate(out, bars, s)
        out_gated_dict[s] = out_g

        proxy_g = summarize_proxy(out_g)

        print(f"\n--- HL STATE {s} ---")
        for k, v in proxy_g.items():
            print(f"{k:>16}: {v}")

    return model, metrics, p_test, out, out_gated_dict

if __name__ == "__main__":
    main()