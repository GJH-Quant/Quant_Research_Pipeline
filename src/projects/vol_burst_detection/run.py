# run.py
from __future__ import annotations

import pathlib
import numpy as np

from .config import (
    # bars
    TIME_INT, BOOL_FILL, RTH_START, RTH_END, PRICE_COL, SIZE_COL,
    # burst
    BURST_FEATURE_COL, TOD_COL,
    PCT_LOOKBACK, PCT_MIN_PERIODS, PCT_Q, PCT_EPS, PCT_OUT_COL,
    MAD_LOOKBACK, MAD_MIN_PERIODS, MAD_FLOOR, MAD_SCALE, MAD_OUT_COL,
    # sanity
    PCT_APPROX_EVENT_SCORE, MAD_EVENT_Z,
)

from .bars import parquet_to_df, create_time_bars
from .features import build_burst_features

# =========================
# RUNTIME INPUTS (keep out of config)
# =========================
TRADES_PATH = pathlib.Path(r"Directory")

DATE_SLICE = ("2025-01-01", "2025-12-20")


def main() -> dict:
    # -------------------------
    # Load trades
    # -------------------------
    trades = parquet_to_df(TRADES_PATH)
    if DATE_SLICE is not None:
        trades = trades.loc[DATE_SLICE[0]:DATE_SLICE[1]]

    # -------------------------
    # Build RTH time bars
    # -------------------------
    bars = create_time_bars(
        trades,
        time_int=TIME_INT,
        bool_fill=BOOL_FILL,
        rth_start=RTH_START,
        rth_end=RTH_END,
        price_col=PRICE_COL,
        size_col=SIZE_COL,
    )

    # -------------------------
    # Build burst features
    # -------------------------
    bars = build_burst_features(
        bars,
        feature_col=BURST_FEATURE_COL,
        time_int=TIME_INT,
        tod_col=TOD_COL,
        pct_lookback=PCT_LOOKBACK,
        pct_quantile=PCT_Q,
        pct_min_periods=PCT_MIN_PERIODS,
        pct_eps=PCT_EPS,
        pct_out_col=PCT_OUT_COL,
        mad_lookback=MAD_LOOKBACK,
        mad_min_periods=MAD_MIN_PERIODS,
        mad_floor=MAD_FLOOR,
        mad_scale=MAD_SCALE,
        mad_out_col=MAD_OUT_COL,
        keep_tod=True,   # set False if you don’t want tod_bin in final bars
    )

    # -------------------------
    # Sanity prints
    # -------------------------
    print("\n" + "=" * 60)
    print(f"BURST DETECTORS ON: {BURST_FEATURE_COL}")
    print("=" * 60)

    pct = bars[PCT_OUT_COL]
    madz = bars[MAD_OUT_COL]

    print(f"[SANITY] {PCT_OUT_COL} non-null: {pct.notna().mean():.2%}")
    if pct.notna().any():
        print(f"[SANITY] {PCT_OUT_COL} quantiles (non-null):")
        print(pct.dropna().quantile([0.5, 0.9, 0.95, 0.99, 0.995]).to_string())
        approx_event_rate = float((pct.dropna() > np.log1p(PCT_APPROX_EVENT_SCORE)).mean())
        print(f"[SANITY] approx pct event rate (score>{PCT_APPROX_EVENT_SCORE}): {approx_event_rate:.4%}")

    print(f"[SANITY] {MAD_OUT_COL} non-null: {madz.notna().mean():.2%}")
    if madz.notna().any():
        print(f"[SANITY] {MAD_OUT_COL} quantiles (non-null):")
        print(madz.dropna().quantile([0.5, 0.9, 0.95, 0.99, 0.995]).to_string())
        approx_mad_event_rate = float((madz.dropna() > MAD_EVENT_Z).mean())
        print(f"[SANITY] approx MAD event rate (z>{MAD_EVENT_Z}): {approx_mad_event_rate:.4%}")

    return {"bars": bars}


if __name__ == "__main__":
    main()
