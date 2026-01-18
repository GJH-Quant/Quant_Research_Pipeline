# ==============================
# MED/MAD TOD NORMALIZATION
# ==============================

import pandas as pd
import numpy as np

def tod_normalization(
    bars: pd.DataFrame,
    col: str = "volume",
    tod_int: str = "5min",
    rth_start: str = "09:30:00",
):

    out = bars.copy()

    rth_open = out.index.normalize() + pd.to_timedelta(rth_start)
    seconds_since_open = (out.index - rth_open).total_seconds()
    tod_step = pd.to_timedelta(tod_int).total_seconds()
    out["tod_bin"] = (seconds_since_open // tod_step).astype("int64")

    stats = out.groupby("tod_bin")[col].agg(
        tod_median="median",
        tod_mad=lambda x: np.median(np.abs(x - np.median(x))),
    )

    out = out.join(stats, on="tod_bin")
    out[f"{col}_tod_z"] = (out[col] - out["tod_median"]) / out["tod_mad"].replace(0, np.nan)

    return out.drop(columns=["tod_bin", "tod_median", "tod_mad"])