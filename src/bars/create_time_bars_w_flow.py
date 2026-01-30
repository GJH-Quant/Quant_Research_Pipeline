# ===================================
# TRADES DATAFRAME ---> TIME BARS
# ===================================

import pandas as pd
import numpy as np

def create_time_bars_w_flow(
    df: pd.DataFrame,
    time_int: str,
    bool_fill: bool,
    rth_start: str,
    rth_end: str,
    price_col: str,
    size_col: str,
    side_col: str,
):

    x = df[[price_col, size_col, side_col]].copy()
    x = x.dropna(subset=[price_col, size_col]).sort_index()

    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    x[size_col]  = pd.to_numeric(x[size_col],  errors="coerce")
    x = x.dropna(subset=[price_col, size_col])

    session_key = x.index.normalize()

    def one_session(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        day = data.index[0].normalize()
        rth_anchor = day + pd.to_timedelta(rth_start)

        data = data.between_time(rth_start, rth_end, inclusive="left")
        if data.empty:
            return pd.DataFrame()


        side = data[side_col].astype("string").str.strip().str.upper()
        is_buy  = side.eq("B")
        is_sell = side.eq("A")
        is_neut = side.eq("N")

        sz = data[size_col].astype("float64")
        px = data[price_col].astype("float64")

        data = data.copy()
        data["_buy_sz"]     = sz * is_buy.astype("int8")
        data["_sell_sz"]    = sz * is_sell.astype("int8")
        data["_neutral_sz"] = sz * is_neut.astype("int8")
        data["_dollar"]     = px * sz


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
            dollar=("_dollar", "sum"),
            aggr_buy_vol=("_buy_sz", "sum"),
            aggr_sell_vol=("_sell_sz", "sum"),
            neutral_vol=("_neutral_sz", "sum"),
        ).asfreq(time_int)

        bars["session_date"] = bars.index.normalize().date


        if bool_fill:

            bars["close"] = bars["close"].ffill()

            for c in ["volume", "dollar", "aggr_buy_vol", "aggr_sell_vol", "neutral_vol"]:
                bars[c] = bars[c].fillna(0.0)

            bars["n_trades"] = bars["n_trades"].fillna(0).astype("int64")

            bars = bars.dropna(subset=["close"])

            bars["inactive"] = bars["n_trades"].eq(0)

            c = bars.loc[bars["inactive"], "close"]
            bars.loc[bars["inactive"], ["open", "high", "low"]] = np.column_stack([c, c, c])

            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])

        else:

            bars = bars.dropna(subset=["close"])
            bars["inactive"] = bars["n_trades"].fillna(0).eq(0)

        eps = 1e-12
        aggr_total = (bars["aggr_buy_vol"] + bars["aggr_sell_vol"] + bars["neutral_vol"]).astype("float64")

        bars["aggr_total"] = aggr_total
        bars["aggr_buy_frac"]  = bars["aggr_buy_vol"]  / (aggr_total + eps)
        bars["aggr_sell_frac"] = bars["aggr_sell_vol"] / (aggr_total + eps)
        bars["neutral_frac"]   = bars["neutral_vol"]   / (aggr_total + eps)

        bars["net_aggr"] = (bars["aggr_buy_vol"] - bars["aggr_sell_vol"]).astype("float64")
        bars["net_aggr_frac"] = bars["net_aggr"] / (aggr_total + eps)

        bars["ret"] = bars["close"].pct_change()
        bars["log_ret"] = np.log1p(bars["ret"])

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