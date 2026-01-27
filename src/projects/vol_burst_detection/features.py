# features.py
from __future__ import annotations
import numpy as np
import pandas as pd

# =========================
# FEATURES (VOL PROXIES)
# =========================
def add_tod_bin(
    bars: pd.DataFrame,
    time_int: str,
    col_name: str,
) -> pd.DataFrame:

    out = bars.copy()
    idx = out.index

    step_sec = int(pd.Timedelta(time_int).total_seconds())
    if step_sec <= 0:
        raise ValueError("time_int must be positive")

    sec = idx.hour * 3600 + idx.minute * 60 + idx.second
    out[col_name] = (sec // step_sec).astype("int32")
    
    return out


def add_rolling_percentile_burst(
    bars: pd.DataFrame,
    feature_col: str,
    tod_col: str,
    lookback: int,
    quantile: float,
    min_periods: int,
    eps: float,
    out_col: str,
) -> pd.DataFrame:

    out = bars.copy()

    x = out[feature_col].astype("float64").replace([np.inf, -np.inf], np.nan)
    x_hist = x.groupby(out[tod_col], sort=False).shift(1)

    thr = (
        x_hist.groupby(out[tod_col], sort=False)
        .rolling(window=lookback, min_periods=min_periods)
        .quantile(quantile)
        .reset_index(level=0, drop=True)
    ).reindex(out.index)

    score = x / thr.clip(lower=eps)
    score = score.where(thr.notna())

    out[out_col] = np.log1p(score)

    return out


def add_med_mad_norm(
    bars: pd.DataFrame,
    feature_col: str,
    tod_col: str,
    lookback: int,
    min_periods: int,
    mad_floor: float,
    scale_mad: float,
    out_col: str,
) -> pd.DataFrame:

    out = bars.copy()

    x = out[feature_col].astype("float64").replace([np.inf, -np.inf], np.nan)
    x_hist = x.groupby(out[tod_col], sort=False).shift(1)

    med = x_hist.groupby(out[tod_col], sort=False).transform(
        lambda s: s.rolling(window=lookback, min_periods=min_periods).median()
    )

    abs_dev = (x_hist - med).abs()

    mad = abs_dev.groupby(out[tod_col], sort=False).transform(
        lambda s: s.rolling(window=lookback, min_periods=min_periods).median()
    )

    denom = (mad * scale_mad).clip(lower=mad_floor)

    z = (x - med) / denom
    z = z.where(med.notna() & mad.notna())

    out[out_col] = z

    return out


def build_burst_features(
    bars: pd.DataFrame,
    feature_col: str,
    time_int: str,
    *,
    tod_col: str,
    # percentile
    pct_lookback: int,
    pct_quantile: float,
    pct_min_periods: int,
    pct_eps: float,
    pct_out_col: str,
    # mad
    mad_lookback: int,
    mad_min_periods: int,
    mad_floor: float,
    mad_scale: float,
    mad_out_col: str,
    keep_tod: bool = True,
) -> pd.DataFrame:
    out = bars.copy()

    out = add_tod_bin(out, time_int=time_int, col_name=tod_col)

    out = add_rolling_percentile_burst(
        out,
        feature_col=feature_col,
        tod_col=tod_col,
        lookback=pct_lookback,
        quantile=pct_quantile,
        min_periods=pct_min_periods,
        eps=pct_eps,
        out_col=pct_out_col,
    )

    out = add_med_mad_norm(
        out,
        feature_col=feature_col,
        tod_col=tod_col,
        lookback=mad_lookback,
        min_periods=mad_min_periods,
        mad_floor=mad_floor,
        scale_mad=mad_scale,
        out_col=mad_out_col,
    )

    if not keep_tod:
        out = out.drop(columns=[tod_col], errors="ignore")

    return out

