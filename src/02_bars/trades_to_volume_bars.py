# ====================================
# TRADES DATAFRAME ---> VOLUME BARS
# ====================================

import pandas as pd
import numpy as np
import pathlib

def create_volume_bars(
    df: pd.DataFrame,
    shares_per_bar: float,
    rth_start: str = "09:30:00",
    rth_end: str = "16:00:00",
    price_col: str = "price",
    size_col: str = "size",
):

    if shares_per_bar <= 0:
        raise ValueError("shares_per_bar must be > 0")

    x = df[[price_col, size_col]].copy()
    x = x.dropna(subset=[price_col, size_col]).sort_index()

    session_key = x.index.normalize()

    def one_session(data: pd.DataFrame) -> pd.DataFrame:

        data = data.between_time(rth_start, rth_end, inclusive="left")
        if data.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume","n_trades","session_date","inactive"])

        # -------------------
        # Build volume bars 
        # -------------------
        v = data[size_col].to_numpy(dtype="float64", copy=False)
        cumv = np.cumsum(v)

        bar_id = np.floor((cumv - 1e-12) / shares_per_bar).astype(np.int64)

        g = data.copy()
        g["_bar_id"] = bar_id

        bars = g.groupby("_bar_id", sort=True).agg(
            open=(price_col, "first"),
            high=(price_col, "max"),
            low=(price_col, "min"),
            close=(price_col, "last"),
            volume=(size_col, "sum"),
            n_trades=(price_col, "count"),
        )
        bar_start = g.reset_index().groupby("_bar_id")["ts_event"].first()
        bars.index = pd.DatetimeIndex(bar_start.to_numpy(), name="bar_start")

        bars["session_date"] = bars.index.normalize().date
        bars["inactive"] = False

        return bars

    bars = x.groupby(session_key, group_keys=False).apply(one_session)

    # -------------------------
    # Sanity prints
    # -------------------------
    label = "filled"
    bars_per_day = bars.groupby(bars.index.normalize()).size()
    print(f"[SANITY] bars/day summary ({label}):")
    print(bars_per_day.describe())

    rth_start_t = pd.to_timedelta(rth_start)
    rth_end_t = pd.to_timedelta(rth_end)
    bar_times = bars.index - bars.index.normalize()
    outside_rth = (bar_times < rth_start_t) | (bar_times >= rth_end_t)

    if outside_rth.any():
        print(f"[SANITY] WARNING: {int(outside_rth.sum())} bars outside RTH")
        print(bars.loc[outside_rth].head())
    else:
        print("[SANITY] OK: no bars outside RTH")

    print(f"[SANITY] inactive rate: {bars['inactive'].mean():.2%}")
    frac_h_eq_l = float((bars["high"] == bars["low"]).mean())
    print(f"[SANITY] fraction(high==low): {frac_h_eq_l:.2%}")

    return bars