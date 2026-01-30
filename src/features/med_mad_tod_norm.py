# ==============================
# MED/MAD TOD NORMALIZATION
# ==============================

import pandas as pd
import numpy as np

def add_med_mad_norm(
    bars: pd.DataFrame,
    feature_col: str,
    bucket_td: str = "60s",
    time_int_td: str = "2s",
    rth_start: str = "09:30:00",
    rth_end: str = "16:00:00",
    lookback_days: int = 10,
    mad_floor: float = 1e-6,
    scale_mad: float = 1.4826,
    verbose: bool = True,
):

    out = bars.copy()
    zcol = f"{feature_col}_mad_z"

    bucket_sec = pd.to_timedelta(bucket_td).total_seconds()
    time_int_sec = pd.to_timedelta(time_int_td).total_seconds()
    rth_start_td = pd.to_timedelta(rth_start)

    secs_from_open = (
        (out.index - out.index.normalize()) - rth_start_td
    ).total_seconds()

    out["tod_bucket"] = (secs_from_open // bucket_sec).astype("int32")

    samples_per_bucket_per_day = int(bucket_sec // time_int_sec)
    lookback = lookback_days * samples_per_bucket_per_day

    x = pd.to_numeric(out[feature_col], errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)
    out["_x"] = x

    g = out.groupby("tod_bucket", sort=False)["_x"]
    x_hist = g.shift(1)

    med = x_hist.groupby(out["tod_bucket"], sort=False).transform(
        lambda s: s.rolling(window=lookback, min_periods=lookback).median()
    )

    mad = (x_hist - med).abs().groupby(out["tod_bucket"], sort=False).transform(
        lambda s: s.rolling(window=lookback, min_periods=lookback).median()
    )

    denom = (mad * scale_mad).clip(lower=mad_floor)

    z = (out["_x"] - med) / denom
    z = z.where(med.notna() & mad.notna())

    out[f'{feature_col}_mad_z'] = z
    out.drop(columns="_x", inplace=True)

    # =========================
    # SANITY PRINTS
    # =========================
    if verbose:
        print("\n==============================")
        print(f"MED/MAD NORMALIZATION CHECK: {feature_col}")
        print("==============================")
        print("Rows:", len(out))
        print("NaN % raw:", out[feature_col].isna().mean())
        print("NaN % z  :", out[zcol].isna().mean())

        print("\nRaw feature stats:")
        print(out[feature_col].describe(percentiles=[.01, .05, .5, .95, .99]))

        print("\nZ-score stats (non-NaN):")
        print(out[zcol].dropna().describe(percentiles=[.01, .05, .5, .95, .99]))

        # extra PM-grade checks
        print("\nZ-score median (should be ~0):", out[zcol].median())
        print("Z-score 99.9%:", out[zcol].quantile(0.999))
        print("Z-score max:", out[zcol].max())

    return out