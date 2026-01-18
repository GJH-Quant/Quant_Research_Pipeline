# ==============================
# TRADES DATAFRAME ---> TIME BARS
# ==============================

import pandas as pd
import numpy as np
import pathlib

def create_time_bars(
    df: pd.DataFrame,
    time_int: str,
    bool_fill: bool,
    rth_start: str = "09:30:00",
    price_col: str = "price",
    size_col: str = "size",
):

    x = df[[price_col, size_col]].copy()
    x = x.dropna(subset=[price_col, size_col]).sort_index()

    session_key = x.index.normalize()

    def one_session(data: pd.DataFrame) -> pd.DataFrame:
        rth_anchor = data.index[0].normalize() + pd.to_timedelta(rth_start)

        # -------------------------
        # Core OHLCV + trades
        # -------------------------
        bars = data.resample(
            time_int,
            origin=rth_anchor,
            label="left",
            closed="left",
        ).agg(
            open=(price_col, "first"),
            high=(price_col, "max"),
            low=(price_col, "min"),
            close=(price_col, "last"),
            volume=(size_col, "sum"),
            n_trades=(price_col, "count"),
        ).asfreq(time_int)

        bars["session_date"] = bars.index.normalize().date

        # -------------------------
        # Fill / inactive logic
        # -------------------------
        if bool_fill:
            bars["close"] = bars["close"].ffill()
            bars["volume"] = bars["volume"].fillna(0.0)
            bars["n_trades"] = bars["n_trades"].fillna(0).astype("int64")

            bars = bars.dropna(subset=["close"])

            bars["inactive"] = bars["n_trades"] == 0

            c = bars.loc[bars["inactive"], "close"]
            bars.loc[bars["inactive"], ["open", "high", "low"]] = np.column_stack([c, c, c])
            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])

        else:
            bars = bars.dropna(subset=["close"])
            bars["inactive"] = bars["n_trades"].fillna(0).eq(0)

        return bars

    bars = x.groupby(session_key, group_keys=False).apply(one_session)

    # -------------------------
    # Sanity prints
    # -------------------------
    bars_per_day = bars.groupby(bars.index.normalize()).size()
    label = "filled" if bool_fill else "unfilled"
    print(f"[SANITY] bars/day summary ({label}):")
    print(bars_per_day.describe())

    rth_start_t = pd.to_timedelta(rth_start)
    rth_end_t = pd.to_timedelta("16:00:00")
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