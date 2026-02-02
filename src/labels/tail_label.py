# ==============================
# TAIL LABELS
# ==============================

import pandas as pd
import numpy as np

def build_tail_labels(
    bars,
    horizon,
    spread_mult=1.0,
    mid_col="mid",
    spread_col="spread",
):
    
    mid = bars[mid_col]
    spr0 = bars[spread_col]
    mid_fwd = mid.shift(-horizon)
    spr_fwd = spr0.shift(-horizon)

    move = mid_fwd - mid
    cost = spread_mult * (spr0 + spr_fwd)

    y_up = (move > cost).astype("int8")
    y_down = (-move > cost).astype("int8")

    valid = mid_fwd.notna() & spr_fwd.notna()
    y_up = y_up.where(valid)
    y_down = y_down.where(valid)

    out = bars.copy()
    out["y_up"] = y_up
    out["y_down"] = y_down

    return out

