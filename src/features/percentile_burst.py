# ==============================
# TOD ROLLING PERCENTILE (BURST)  [NO QUANTILE VERSION]
# ==============================

import pandas as pd
import numpy as np

def add_rolling_percentile_burst(
    bars: pd.DataFrame,
    feature_col: str,
    bucket_td: str = "60s",
    time_int_td: str = "2s",
    rth_start: str = "09:30:00",
    rth_end: str = "16:00:00",
    lookback_days: int = 10,
    eps: float = 1e-12,
    verbose: bool = True,
):

    out = bars.copy()
    out_col = f"{feature_col}_pct_score"


    bucket_sec = pd.to_timedelta(bucket_td).total_seconds()
    rth_start_td = pd.to_timedelta(rth_start)
    rth_end_td = pd.to_timedelta(rth_end)

    secs_from_open = ((out.index - out.index.normalize()) - rth_start_td).total_seconds()
    out["tod_bucket"] = (secs_from_open // bucket_sec).astype("int32")

    in_rth = (secs_from_open >= 0) & (secs_from_open < (rth_end_td - rth_start_td).total_seconds())

    x = pd.to_numeric(out[feature_col], errors="coerce").astype("float64")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.where(in_rth)
    out["_x"] = x

    time_int_sec = pd.to_timedelta(time_int_td).total_seconds()
    samples_per_bucket_per_day = int(bucket_sec // time_int_sec)
    lookback = lookback_days * samples_per_bucket_per_day

    g = out.groupby("tod_bucket", sort=False)["_x"]
    x_hist = g.shift(1)

    thr = x_hist.groupby(out["tod_bucket"], sort=False).transform(
        lambda s: s.rolling(window=lookback, min_periods=lookback).median()
    )

    denom = thr.clip(lower=eps)
    score = out["_x"] / denom
    score = score.where(thr.notna() & out["_x"].notna())

    out[out_col] = np.log1p(score)
    out.drop(columns="_x", inplace=True)

    # =========================
    # SANITY PRINTS
    # =========================
    if verbose:
        print("\n==============================")
        print(f"TOD BURST CHECK (NO QUANTILE): {feature_col} -> {out_col}")
        print("==============================")
        print("Rows:", len(out))
        print("NaN % raw :", out[feature_col].isna().mean())
        print("NaN % thr :", thr.isna().mean())
        print("NaN % score:", out[out_col].isna().mean())

        print("\nRaw feature stats:")
        print(out[feature_col].describe(percentiles=[.01, .05, .5, .95, .99]))

        print("\nBaseline stats (non-NaN):")
        print(pd.Series(thr, index=out.index).dropna().describe(percentiles=[.01, .05, .5, .95, .99]))

        print("\nScore stats (non-NaN):")
        print(out[out_col].dropna().describe(percentiles=[.01, .05, .5, .95, .99]))

        print("\nScore median:", out[out_col].median())
        print("Score 99.9%:", out[out_col].quantile(0.999))
        print("Score max  :", out[out_col].max())

    return out