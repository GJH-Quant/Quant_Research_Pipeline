# bars.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pathlib

# =========================
# PARQUET --> DATAFRAME
# =========================
def parquet_to_df(in_path: pathlib.Path) -> pd.DataFrame:

    paths = sorted(in_path.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in: {in_path}")

    dfs = [pd.read_parquet(p) for p in paths]
    out = pd.concat(dfs, axis=0).sort_index()

    return out

# =========================
# TIME BARS (RTH-anchored)
# =========================
def create_time_bars(
    df: pd.DataFrame,
    time_int: str,
    bool_fill: bool,
    rth_start: str,
    rth_end: str,
    price_col: str,
    size_col: str,
) -> pd.DataFrame:

    if df.empty:
        print("[SANITY] WARNING: input trades df is empty.")
        return pd.DataFrame()

    x = df[[price_col, size_col]].copy()
    x[price_col] = x[price_col].astype("float64")

    session_key = x.index.normalize()

    def one_session(data: pd.DataFrame) -> pd.DataFrame:
        data = data.between_time(rth_start, rth_end, inclusive="left")
        if data.empty:
            return pd.DataFrame()

        rth_anchor = data.index[0].normalize() + pd.to_timedelta(rth_start)

        bars = (
            data.resample(
                time_int,
                origin=rth_anchor,
                label="left",
                closed="left",
            )
            .agg(
                open=(price_col, "first"),
                high=(price_col, "max"),
                low=(price_col, "min"),
                close=(price_col, "last"),
                volume=(size_col, "sum"),
                n_trades=(price_col, "count"),
            )
            .asfreq(time_int)
        )

        bars["dollar"] = (
            (data[price_col] * data[size_col])
            .resample(
                time_int,
                origin=rth_anchor,
                label="left",
                closed="left",
            )
            .sum()
            .asfreq(time_int)
        )

        bars["session_date"] = bars.index.normalize().date

        if bool_fill:
            bars["close"] = bars["close"].ffill()
            bars["volume"] = bars["volume"].fillna(0)
            bars["n_trades"] = bars["n_trades"].fillna(0).astype("int64")
            bars["dollar"] = bars["dollar"].fillna(0)

            bars = bars.dropna(subset=["close"])
            bars["inactive"] = bars["n_trades"] == 0

            c = bars.loc[bars["inactive"], "close"]
            bars.loc[bars["inactive"], ["open", "high", "low"]] = np.column_stack([c, c, c])
            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])
        else:
            bars = bars.dropna(subset=["close"])
            bars["inactive"] = False

        bars['ret'] = bars['close'].pct_change(fill_method=None)
        bars['log_ret'] = np.log(bars['close']).diff()

        return bars

    bars = x.groupby(session_key, group_keys=False).apply(one_session)

    # ---- HARD GUARD: empty output ----
    if bars.empty:
        print("[SANITY] WARNING: no bars were produced (bars is empty).")
        print("         Likely no trades after symbol/date/RTH filters.")
        return bars

    # --- sanity prints ---
    bars_per_day = bars.groupby(bars.index.normalize()).size()
    label = "filled" if bool_fill else "unfilled"
    print(f"[SANITY] bars/day summary ({label}):")
    print(bars_per_day.describe())

    rth_start_t = pd.to_timedelta(rth_start)
    rth_end_t = pd.to_timedelta(rth_end)
    bar_times = bars.index - bars.index.normalize()
    outside_rth = (bar_times < rth_start_t) | (bar_times >= rth_end_t)
    n_outside = int(outside_rth.sum())

    if n_outside:
        print(f"[SANITY] WARNING: {n_outside} bars outside RTH")
        print(bars.loc[outside_rth].head(5))
    else:
        print("[SANITY] OK: no bars outside RTH")

    if "inactive" in bars.columns:
        print(f"[SANITY] inactive rate: {bars['inactive'].mean():.2%}")
    frac_h_eq_l = float((bars["high"] == bars["low"]).mean())
    print(f"[SANITY] fraction(high==low): {frac_h_eq_l:.2%}")

    print(bars.isna().sum())

    return bars