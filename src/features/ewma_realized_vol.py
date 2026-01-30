# ==============================
# EWMA REALIZED VOL
# ==============================

import numpy as np
import pandas as pd

def add_ewma_realized_vol(
    bars: pd.DataFrame,
    logret_col: str = "log_ret",
    session_col: str = "session_date",
    span: int = 60,
    min_periods: int = 20,
    out_col: str = "rv_ewma",
) -> pd.DataFrame:
    out = bars.copy()

    r = pd.to_numeric(out[logret_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rv = r.pow(2)

    out[out_col] = (
        rv.groupby(out[session_col], sort=False)
          .transform(lambda s: np.sqrt(s.ewm(span=span, adjust=False, min_periods=min_periods).mean()))
    )

    return out