# ==============================
# LOG HIGH/LOW VOLATILITY
# ==============================

import pandas as pd
import numpy as np

def add_log_hl_vol(
    bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
):
    
    out = bars.copy()
    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")

    out["log_hl"] = np.log(h / l)
    out["log_hl"] = out["log_hl"].replace([np.inf, -np.inf], np.nan)
    
    return out