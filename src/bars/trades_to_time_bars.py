# =============================================================
#  TRADES TO TIME BARS (MULTI-DAY, PER-SESSION; NO OVERNIGHT)
# =============================================================

import pandas as pd
import numpy as np
import pathlib

def create_trades_time_bars(df: pd.DataFrame, time_int: str, bool_fill: bool) -> pd.DataFrame:

    # --------------------------
    # Input validation + cleaning
    # --------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    x = df[["price", "size"]].copy().sort_index()
    x = x.loc['2024-12-25':].copy()

    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"]  = pd.to_numeric(x["size"], errors="coerce")
    x = x.dropna(subset=["price", "size"])
    x = x[x["size"] > 0]

    if len(x) == 0:
        print("No valid trades after cleaning.")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "active"])

    tz = x.index.tz
    session_key = x.index.normalize() if tz is not None else x.index.date

    # --------------------------
    # Per-session builder
    # --------------------------
    def _one_session(d: pd.DataFrame) -> pd.DataFrame:
        bars = d.resample(time_int).agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
        )

        # Full grid within this session span (no overnight because we do this per-session)
        bars = bars.asfreq(time_int)

        if bool_fill:
            # 1) Close forward-fill
            bars["close"] = bars["close"].ffill()

            # 2) Volume to 0 for no-trade bars
            bars["volume"] = bars["volume"].fillna(0)

            # 3) Force OHLC = close for no-trade bars (bulletproof)
            inactive = bars["volume"].eq(0) & bars["close"].notna()
            c = bars.loc[inactive, "close"]
            bars.loc[inactive, "open"] = c
            bars.loc[inactive, "high"] = c
            bars.loc[inactive, "low"]  = c

            # 4) Defensive: fill any remaining OHLC NaNs from close
            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])

        # Active flag
        bars["active"] = bars["volume"].fillna(0) > 0
        return bars

    # Build multi-day bars
    bars = x.groupby(session_key, group_keys=False).apply(_one_session)

    # ==========================================================
    # SANITY CHECKS (PRINT ONLY)
    # ==========================================================

    # 1) Overnight bars (should be zero)
    overnight = bars.between_time("16:00", "09:30", inclusive="neither")
    print(f"[Sanity] Overnight bars: {len(overnight)}")

    # 2) Volume conservation
    orig_vol = x["size"].sum()
    bar_vol  = bars["volume"].sum()
    print(f"[Sanity] Volume original={orig_vol:.0f}  bars={bar_vol:.0f}")

    # 3) Inactive bars should have flat prices (ignore rows with NaN close)
    inactive_df = bars[~bars["active"]]
    if len(inactive_df) > 0:
        inactive_df2 = inactive_df[inactive_df["close"].notna()]
        bad_flat = inactive_df2[
            (inactive_df2["open"] != inactive_df2["close"]) |
            (inactive_df2["high"] != inactive_df2["close"]) |
            (inactive_df2["low"]  != inactive_df2["close"])
        ]
        print(f"[Sanity] Inactive bars with non-flat price: {len(bad_flat)}")
    else:
        print("[Sanity] No inactive bars")

    # 4) Zero returns on inactive bars (ignore NaN returns)
    returns = bars["close"].pct_change()
    bad_ret = returns[(bars["active"] == False) & (returns.notna()) & (returns != 0)]
    print(f"[Sanity] Non-zero returns on inactive bars: {len(bad_ret)}")

    # 5) Bars per session (spot anomalies)
    bars_per_day = bars.groupby(bars.index.normalize()).size()
    print(
        "[Sanity] Bars/day: "
        f"min={bars_per_day.min()} "
        f"median={bars_per_day.median()} "
        f"max={bars_per_day.max()}"
    )

    # Debug peek
    inactive_peek = bars[~bars["active"]][["open", "high", "low", "close", "volume"]]
    print(inactive_peek.head(20))
    print("NaN close on inactive:", inactive_peek["close"].isna().sum())

    return bars

    