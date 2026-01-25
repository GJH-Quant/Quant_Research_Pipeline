# =========================
# FILE: ev.py
# =========================
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def ensure_session_date(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "session_date" not in x.columns:
        x["session_date"] = x.index.normalize()
    return x


def add_forward_quotes_H(df: pd.DataFrame, H: int) -> pd.DataFrame:
    x = ensure_session_date(df)
    x["bid_px_00"] = pd.to_numeric(x["bid_px_00"], errors="coerce")
    x["ask_px_00"] = pd.to_numeric(x["ask_px_00"], errors="coerce")

    x["spread_t"] = (x["ask_px_00"] - x["bid_px_00"]).astype("float64")
    x["mid_t"] = 0.5 * (x["bid_px_00"] + x["ask_px_00"])

    g = x.groupby("session_date", group_keys=False)
    x["bid_tH"] = g["bid_px_00"].shift(-H)
    x["mid_tH"] = g["mid_t"].shift(-H)
    x["can_eval_H"] = x["bid_tH"].notna() & x["mid_tH"].notna()
    return x


def compute_ev_variants(df: pd.DataFrame, H: int, cost_bps: float = 0.0) -> pd.DataFrame:
    x = add_forward_quotes_H(df, H=H)
    valid = x["can_quote"].astype(bool) & x["can_eval_H"].astype(bool) & x["bid_px_00"].notna()
    if valid.sum() == 0:
        return pd.DataFrame([{"variant": "none", "n_valid_quotes": 0}])

    filled = x["fill_bid_H"].astype(np.int8)
    f = filled[valid].to_numpy(dtype=float)

    entry = x.loc[valid, "bid_px_00"].to_numpy(dtype=float)
    exit_bid = x.loc[valid, "bid_tH"].to_numpy(dtype=float)
    exit_mid = x.loc[valid, "mid_tH"].to_numpy(dtype=float)
    cost = entry * (float(cost_bps) / 1e4)

    def summarize(name: str, exit_px: np.ndarray) -> Dict[str, float]:
        pnl = (exit_px - entry) - cost
        ev_q = float((f * pnl).mean())
        ev_q_bps = float(((f * pnl) / entry).mean() * 1e4)
        fill_rate = float(f.mean())

        mask_fill = (filled == 1) & valid
        if mask_fill.any():
            e_f = x.loc[mask_fill, "bid_px_00"].to_numpy(dtype=float)
            x_f = (x.loc[mask_fill, "bid_tH"] if name == "exit_bid" else x.loc[mask_fill, "mid_tH"]).to_numpy(dtype=float)
            c_f = e_f * (float(cost_bps) / 1e4)
            pnl_f = (x_f - e_f) - c_f
            ev_f = float(pnl_f.mean())
            ev_f_bps = float((pnl_f / e_f).mean() * 1e4)
        else:
            ev_f, ev_f_bps = np.nan, np.nan

        return {
            "variant": name,
            "fill_rate_on_valid": fill_rate,
            "ev_per_quote_$": ev_q,
            "ev_per_quote_bps": ev_q_bps,
            "ev_per_fill_$": ev_f,
            "ev_per_fill_bps": ev_f_bps,
        }

    res = pd.DataFrame([summarize("exit_bid", exit_bid), summarize("exit_mid", exit_mid)])
    res["valid_rate"] = float(valid.mean())
    res["n_valid_quotes"] = int(valid.sum())
    return res
