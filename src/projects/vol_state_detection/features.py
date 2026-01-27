from __future__ import annotations
import numpy as np
import pandas as pd

# =========================
# FEATURES (VOL PROXIES)
# =========================

def add_log_hl_vol(
    bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    
    out = bars.copy()

    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")

    out["log_hl"] = np.log(h / l)
    out["log_hl"] = out["log_hl"].replace([np.inf, -np.inf], np.nan)

    return out

def add_ewma_realized_vol(
    bars: pd.DataFrame,
    close_col: str = "close",
    span: int = 30,
    min_periods: int = 10,
) -> pd.DataFrame:

    out = bars.copy()

    out["rv_ewma"] = (
        out["log_ret"]
        .pow(2)
        .groupby(out["session_date"], sort=False)
        .transform(lambda s: np.sqrt(s.ewm(span=span, adjust=False, min_periods=min_periods).mean()))
)

    return out