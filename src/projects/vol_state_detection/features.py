from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_hl_vol(
    bars: pd.DataFrame,
    high_col: str,
    low_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = bars.copy()

    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")

    x = np.log(h / l)
    x = pd.Series(x, index=out.index).replace([np.inf, -np.inf], np.nan)

    out[out_col] = x
    return out


def add_ewma_realized_vol(
    bars: pd.DataFrame,
    logret_col: str,
    session_col: str,
    span: int,
    min_periods: int,
    out_col: str,
) -> pd.DataFrame:
    out = bars.copy()

    out[out_col] = (
        out[logret_col]
        .pow(2)
        .groupby(out[session_col], sort=False)
        .transform(lambda s: np.sqrt(s.ewm(span=span, adjust=False, min_periods=min_periods).mean()))
    )

    return out


def build_vol_features(
    bars: pd.DataFrame,
    high_col: str,
    low_col: str,
    log_hl_out_col: str,
    logret_col: str,
    session_col: str,
    rv_span: int,
    rv_min_periods: int,
    rv_out_col: str,
) -> pd.DataFrame:

    out = bars.copy()

    out = add_log_hl_vol(out, high_col=high_col, low_col=low_col, out_col=log_hl_out_col)

    out = add_ewma_realized_vol(
        out,
        logret_col=logret_col,
        session_col=session_col,
        span=rv_span,
        min_periods=rv_min_periods,
        out_col=rv_out_col,
    )

    return out
