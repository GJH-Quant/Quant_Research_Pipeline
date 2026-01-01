import pandas as pd
import numpy as np
import pathlib

def create_trades_volume_bars(
    df: pd.DataFrame,
    shares_per_bar: float,
    tz_target: str = "America/New_York",
    rth_start: str = "09:30",
    rth_end: str = "16:00",
) -> pd.DataFrame:

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    x = df[["price", "size"]].copy().sort_index()
    x = x.loc["2024-12-25":].copy()

    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"]  = pd.to_numeric(x["size"], errors="coerce")
    x = x.dropna(subset=["price", "size"])
    x = x.loc[x["size"] > 0]

    if len(x) == 0:
        print("No valid trades after cleanup")
        return pd.DataFrame(columns=["open","high","low","close","volume","dollar","n_trades"])

    idx = x.index
    if idx.tz is None:
        x.index = idx.tz_localize("UTC")
    x.index = x.index.tz_convert(tz_target)

    x = x.between_time(rth_start, rth_end, inclusive="left")

    if len(x) == 0:
        print("No valid RTH trades after filtering.")
        return pd.DataFrame(columns=["open","high","low","close","volume","dollar","n_trades"])

    session_key = x.index.normalize()

    x["dollar"] = x["price"] * x["size"]

    def _one_session(d: pd.DataFrame) -> pd.DataFrame:
        cs = d["size"].cumsum()
        bucket = (cs // float(shares_per_bar)).astype("int64")

        g = d.groupby(bucket, sort=True)

        out = g.agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            dollar=("dollar", "sum"),
            n_trades=("price", "count"),
        )

        ts_end = g["price"].apply(lambda s: s.index[-1])
        out.index = pd.DatetimeIndex(ts_end)
        out.index.name = "ts_event"
        return out

    bars = x.groupby(session_key, group_keys=False).apply(_one_session).sort_index()

    if len(bars) == 0:
        print("[Sanity Check] No bars created.")
        return bars

    orig_vol = x["size"].sum()
    bar_vol  = bars["volume"].sum()
    print(f"[Sanity Check] Volume original={orig_vol:.0f}  bars={bar_vol:.0f}")

    print(
        f"[Sanity Check] shares_per_bar={float(shares_per_bar):.2f}  "
        f"bar_volume: min={bars['volume'].min():.0f}  "
        f"median={bars['volume'].median():.0f}  "
        f"max={bars['volume'].max():.0f}"
    )

    ts_local = bars.index.tz_convert(tz_target) if bars.index.tz is not None else bars.index
    end_times = ts_local.time
    rth_start_t = pd.Timestamp(rth_start).time()
    rth_end_t   = pd.Timestamp(rth_end).time()
    outside = (end_times < rth_start_t) | (end_times >= rth_end_t)
    print(f"[Sanity Check] Bars ending outside RTH: {int(outside.sum())}")

    return bars