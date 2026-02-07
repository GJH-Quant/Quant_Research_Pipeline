# ==============================
# TAIL LABELS
# ==============================

import pandas as pd
import numpy as np

def build_tail_labels_session_aware(
    bars,
    horizon: int,
    spread_mult: float = 1.0,
    mid_col: str = "mid",
    spread_col: str = "spread",
    session_col: str = "session_date",
):
    g = bars.groupby(session_col, sort=False)

    mid = bars[mid_col]
    spr0 = bars[spread_col]

    mid_fwd = g[mid_col].shift(-horizon)
    spr_fwd = g[spread_col].shift(-horizon)

    move = mid_fwd - mid
    cost = spread_mult * (spr0 + spr_fwd)

    y_up = (move > cost).astype("int8")
    y_down = (-move > cost).astype("int8")

    valid = mid_fwd.notna() & spr_fwd.notna()

    out = bars.copy()
    out["y_up"] = y_up.where(valid)
    out["y_down"] = y_down.where(valid)
    out["two_sided_spread_cost"] = (spr0 + spr_fwd).where(valid)

    return out