# ===================================
# TRADES + BBO DATAFRAMES ---> TIME BARS
# ===================================

import numpy as np
import pandas as pd

def create_time_bars_w_bbo(
    df: pd.DataFrame,
    bbo: pd.DataFrame,
    time_int: str,
    bool_fill: bool,
    rth_start: str = "09:30:00",
    rth_end: str = "16:00:00",
    price_col: str = "price",
    size_col: str = "size",
    side_col: str = "side",
) -> pd.DataFrame:

    x = df[[price_col, size_col, side_col]].copy()
    x = x.dropna(subset=[price_col, size_col]).sort_index()

    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    x[size_col]  = pd.to_numeric(x[size_col],  errors="coerce")
    x = x.dropna(subset=[price_col, size_col])

    session_key = x.index.normalize()

    b = bbo[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].copy().sort_index()
    for c in ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")

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
        data = data.copy()
        data["_buy_sz"]     = sz * is_buy.astype("int8")
        data["_sell_sz"]    = sz * is_sell.astype("int8")
        data["_neutral_sz"] = sz * is_neut.astype("int8")

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
            aggr_buy_vol=("_buy_sz", "sum"),
            aggr_sell_vol=("_sell_sz", "sum"),
            neutral_vol=("_neutral_sz", "sum"),
        ).asfreq(time_int)

        bars["session_date"] = bars.index.normalize().date

        # -------------------------
        # BBO -> same bars
        # -------------------------
        start = rth_anchor
        end = day + pd.to_timedelta(rth_end)
        b_sess = b.loc[(b.index >= start) & (b.index < end)]

        if not b_sess.empty:
            bb = b_sess.copy()

            bb["bar"] = bb.index.floor(time_int)

            b2 = bb.groupby("bar").agg(
                bid=("bid_px_00", "last"),
                ask=("ask_px_00", "last"),
                bid_sz=("bid_sz_00", "last"),
                ask_sz=("ask_sz_00", "last"),

                bid_changes=("bid_px_00", lambda s: max(int(s.nunique()) - 1, 0)),
                ask_changes=("ask_px_00", lambda s: max(int(s.nunique()) - 1, 0)),
                bbo_updates=("bid_px_00", "size"),
            )

            b2 = b2.reindex(bars.index)

            b2["mid"] = (b2["bid"] + b2["ask"]) / 2.0
            b2["spread"] = (b2["ask"] - b2["bid"]).astype("float64")

            denom = (b2["bid_sz"].astype("float64") + b2["ask_sz"].astype("float64"))
            b2["imbalance"] = np.where(
                denom > 0,
                (b2["bid_sz"].astype("float64") - b2["ask_sz"].astype("float64")) / denom,
                np.nan,
            )

            b2["microprice"] = np.where(
                denom > 0,
                (
                    b2["ask"].astype("float64") * b2["bid_sz"].astype("float64")
                    + b2["bid"].astype("float64") * b2["ask_sz"].astype("float64")
                ) / denom,
                np.nan,
            )
            b2["microprice_diff"] = b2["microprice"] - b2["mid"]

            if bool_fill:

                snap_cols = ["bid", "ask", "bid_sz", "ask_sz", "mid", "spread", "imbalance", "microprice", "microprice_diff"]
                b2[snap_cols] = b2[snap_cols].ffill()

                for c in ["bid_changes", "ask_changes", "bbo_updates"]:
                    b2[c] = b2[c].fillna(0).astype("int64")
            else:
                for c in ["bid_changes", "ask_changes", "bbo_updates"]:
                    b2[c] = b2[c].fillna(0).astype("int64")

            bars = bars.join(b2, how="left")

        else:
            snap_cols = ["bid", "ask", "bid_sz", "ask_sz", "mid", "spread", "imbalance", "microprice", "microprice_diff"]
            for c in snap_cols:
                bars[c] = np.nan
            for c in ["bid_changes", "ask_changes", "bbo_updates"]:
                bars[c] = 0

        if bool_fill:
            bars["close"] = bars["close"].ffill()

            bars["volume"] = bars["volume"].fillna(0.0)
            bars["n_trades"] = bars["n_trades"].fillna(0).astype("int64")

            bars["aggr_buy_vol"] = bars["aggr_buy_vol"].fillna(0.0)
            bars["aggr_sell_vol"] = bars["aggr_sell_vol"].fillna(0.0)
            bars["neutral_vol"] = bars["neutral_vol"].fillna(0.0)

            bars = bars.dropna(subset=["close"])

            bars["inactive"] = bars["n_trades"].eq(0)

            c = bars.loc[bars["inactive"], "close"]
            bars.loc[bars["inactive"], ["open", "high", "low"]] = np.column_stack([c, c, c])

            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])
            
        else:
            bars = bars.dropna(subset=["close"])
            bars["inactive"] = bars["n_trades"].fillna(0).eq(0)

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