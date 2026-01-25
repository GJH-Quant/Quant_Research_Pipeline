# =========================
# FILE: data.py
# =========================
from __future__ import annotations

import pathlib
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

# Optional but strongly recommended for chunked parquet reads
try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

import databento as db


# -------------------------
# Small helpers
# -------------------------

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rth_filter(df: pd.DataFrame, rth_start: str, rth_end: str) -> pd.DataFrame:
    return df.between_time(rth_start, rth_end, inclusive="left")


def rth_grid(day: pd.Timestamp, rth_start: str, rth_end: str, freq: str, tz) -> pd.DatetimeIndex:
    return pd.date_range(
        start=day + pd.to_timedelta(rth_start),
        end=day + pd.to_timedelta(rth_end),
        freq=freq,
        tz=tz,
        inclusive="left",
    )


def _timedelta_from_freq(freq: str) -> pd.Timedelta:
    # "1s", "5s", "100ms" etc
    try:
        return pd.Timedelta(freq)
    except Exception as e:
        raise ValueError(f"Invalid freq={freq!r}. Example valid: '1s', '5s'.") from e


# ============================================================
# 1) RAW -> NORMALIZED PARQUETS (file-by-file; no whole dataset)
# ============================================================

def normalize_mbp1_file_to_parquet(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    tz: str,
    rth_start: str,
    rth_end: str,
) -> None:
    MBP1_COLS = ["ts_event", "sequence", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]

    store = db.DBNStore.from_file(in_file)
    df = store.to_df()[MBP1_COLS].copy()

    df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    df = df.set_index("ts_event").tz_convert(tz)
    df = rth_filter(df, rth_start, rth_end)

    df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce").astype("UInt32")
    for c in ("bid_px_00", "ask_px_00"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("bid_sz_00", "ask_sz_00"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("UInt32")

    df = df.loc[
        (df["sequence"] > 0)
        & (df["bid_px_00"] > 0) & (df["ask_px_00"] > 0)
        & (df["bid_sz_00"] > 0) & (df["ask_sz_00"] > 0)
    ]
    df = df.loc[df["bid_px_00"] <= df["ask_px_00"]]

    df = (
        df.reset_index()
          .sort_values(["ts_event", "sequence"], kind="mergesort")
          .set_index("ts_event")
    )
    df.index.name = "ts_event"

    ensure_dir(out_file.parent)
    df.to_parquet(out_file)


def normalize_trades_file_to_parquet(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    tz: str,
    rth_start: str,
    rth_end: str,
) -> None:
    store = db.DBNStore.from_file(in_file)
    df = store.to_df()[["ts_event", "side", "price", "size", "sequence", "symbol"]].copy()

    df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    df = df.set_index("ts_event").tz_convert(tz)
    df = rth_filter(df, rth_start, rth_end)

    df["side"] = df["side"].astype("string").str.strip().str.upper().astype("category")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    df["size"] = pd.to_numeric(df["size"], errors="coerce").astype("UInt32")
    df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce").astype("UInt32")

    df = df.loc[(df["price"] > 0) & (df["size"] > 0)]
    df = df.reset_index().sort_values(["ts_event", "sequence"], kind="mergesort").set_index("ts_event")
    df.index.name = "ts_event"

    ensure_dir(out_file.parent)
    df.to_parquet(out_file)


def normalize_raw_dirs(
    raw_trades_dir: pathlib.Path,
    raw_mbp1_dir: pathlib.Path,
    norm_trades_dir: pathlib.Path,
    norm_mbp1_dir: pathlib.Path,
    tz: str,
    rth_start: str,
    rth_end: str,
    *,
    overwrite: bool = True,
) -> Tuple[list[pathlib.Path], list[pathlib.Path]]:
    """
    Converts *.dbn.zst -> normalized parquet, file-by-file.
    Returns lists of written parquet paths (trades, mbp1).
    """
    ensure_dir(norm_trades_dir)
    ensure_dir(norm_mbp1_dir)

    trades_out: list[pathlib.Path] = []
    mbp1_out: list[pathlib.Path] = []

    for f in sorted(raw_mbp1_dir.glob("*.dbn.zst")):
        stem = f.name.removesuffix(".mbp-1.dbn.zst").removeprefix("xnas-itch-")
        out = norm_mbp1_dir / f"{stem}.mbp-1.parquet"
        if out.exists() and not overwrite:
            mbp1_out.append(out)
            continue
        normalize_mbp1_file_to_parquet(f, out, tz=tz, rth_start=rth_start, rth_end=rth_end)
        mbp1_out.append(out)

    for f in sorted(raw_trades_dir.glob("*.dbn.zst")):
        stem = f.name.removesuffix(".trades.dbn.zst").removeprefix("xnas-itch-")
        out = norm_trades_dir / f"{stem}.trades.parquet"
        if out.exists() and not overwrite:
            trades_out.append(out)
            continue
        normalize_trades_file_to_parquet(f, out, tz=tz, rth_start=rth_start, rth_end=rth_end)
        trades_out.append(out)

    return trades_out, mbp1_out


# ============================================================
# 2) NORMALIZED -> freq GRID artifacts (day-by-day, in memory)
# ============================================================

def mbp1_to_grid(
    mbp1: pd.DataFrame,
    *,
    day: pd.Timestamp,
    rth_start: str,
    rth_end: str,
    freq: str,
) -> pd.DataFrame:
    """
    Resample MBP1 to freq with LAST quote in the bin, then ffill on a full RTH grid.
    """
    x = mbp1[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].copy().sort_index()
    tz = x.index.tz

    q = x.resample(freq, label="left", closed="left").last()

    grid = rth_grid(day, rth_start, rth_end, freq=freq, tz=tz)
    q = q.reindex(grid).ffill()

    q = q.dropna(subset=["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
    q.index.name = "ts_event"
    return q


def trades_to_tradebars_grid(
    trades: pd.DataFrame,
    *,
    day: pd.Timestamp,
    rth_start: str,
    rth_end: str,
    freq: str,
) -> pd.DataFrame:
    """
    Build OHLCV + aggressive flow per freq on full RTH grid, per-day only.
    """
    need = ["price", "size", "side"]
    d = trades[need].copy().sort_index()
    tz = d.index.tz

    d = rth_filter(d, rth_start, rth_end)
    if d.empty:
        return pd.DataFrame()

    side = d["side"].astype("string")
    is_buy = side.eq("B")
    is_sell = side.eq("A")
    is_neut = side.eq("N")

    px = pd.to_numeric(d["price"], errors="coerce").astype("float64")
    sz = pd.to_numeric(d["size"], errors="coerce").astype("float64")

    tmp = pd.DataFrame(index=d.index)
    tmp["_px"] = px
    tmp["_sz"] = sz
    tmp["_dollar"] = px * sz
    tmp["_buy_sz"] = sz * is_buy.to_numpy(np.int8)
    tmp["_sell_sz"] = sz * is_sell.to_numpy(np.int8)
    tmp["_neut_sz"] = sz * is_neut.to_numpy(np.int8)

    bars = tmp.resample(freq, label="left", closed="left").agg(
        open=("_px", "first"),
        high=("_px", "max"),
        low=("_px", "min"),
        close=("_px", "last"),
        volume=("_sz", "sum"),
        n_trades=("_px", "count"),
        dollar_vol=("_dollar", "sum"),
        aggr_buy_vol=("_buy_sz", "sum"),
        aggr_sell_vol=("_sell_sz", "sum"),
        neutral_vol=("_neut_sz", "sum"),
    )

    grid = rth_grid(day, rth_start, rth_end, freq=freq, tz=tz)
    bars = bars.reindex(grid)

    bars["close"] = bars["close"].ffill()
    bars = bars.dropna(subset=["close"])

    vol_cols = ["volume", "dollar_vol", "aggr_buy_vol", "aggr_sell_vol", "neutral_vol"]
    for c in vol_cols:
        bars[c] = pd.to_numeric(bars[c], errors="coerce").fillna(0.0)

    bars["n_trades"] = pd.to_numeric(bars["n_trades"], errors="coerce").fillna(0).astype("int64")
    bars["inactive"] = bars["n_trades"].eq(0)

    m = bars["inactive"]
    if m.any():
        bars.loc[m, "open"] = bars.loc[m, "close"]
        bars.loc[m, "high"] = bars.loc[m, "close"]
        bars.loc[m, "low"] = bars.loc[m, "close"]
    bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])

    bars["session_date"] = bars.index.normalize()
    bars["aggr_total_vol"] = bars["aggr_buy_vol"] + bars["aggr_sell_vol"]
    bars["net_aggr_vol"] = bars["aggr_buy_vol"] - bars["aggr_sell_vol"]

    vol = bars["volume"].to_numpy("float64")
    ag = bars["aggr_total_vol"].to_numpy("float64")
    net = bars["net_aggr_vol"].to_numpy("float64")
    bars["aggr_frac"] = np.divide(ag, vol, out=np.zeros_like(ag), where=vol > 0)
    bars["net_aggr_frac"] = np.divide(net, vol, out=np.zeros_like(net), where=vol > 0)

    bars.index.name = "ts_event"
    return bars


# ============================================================
# 3) SELL@BID in a memory-safe way (chunked parquet reads)
# ============================================================

def _parquet_batches(
    path: pathlib.Path,
    columns: list[str],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """
    Yields pandas DataFrames indexed by ts_event.

    Handles both cases:
      - timestamp comes back as a column (ts_event / __index_level_0__ / index)
      - timestamp comes back as the DataFrame index (via pandas metadata)
    """
    if pq is None:
        raise RuntimeError("pyarrow is required for chunked parquet reads. Install pyarrow.")

    pf = pq.ParquetFile(path)

    # Request potential timestamp fields too (harmless if not present in output)
    req_cols = list(dict.fromkeys(columns + ["ts_event", "__index_level_0__", "index"]))

    for rb in pf.iter_batches(batch_size=batch_rows, columns=req_cols):
        df = rb.to_pandas()

        # ---- CASE 1: Arrow already restored the timestamp as the index ----
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = df.index.name or "ts_event"
            # normalize name to 'ts_event' for downstream consistency
            if df.index.name != "ts_event":
                df.index.name = "ts_event"
            yield df
            continue

        # ---- CASE 2: timestamp is present as a column ----
        if "ts_event" in df.columns:
            ts = df.pop("ts_event")
        elif "__index_level_0__" in df.columns:
            ts = df.pop("__index_level_0__")
        elif "index" in df.columns:
            ts = df.pop("index")
        else:
            raise RuntimeError(
                "Chunked parquet read could not find timestamp as index or column.\n"
                f"Columns in batch: {list(df.columns)}\n"
                "Fix: write trades parquet with a ts_event column, or preserve pandas index metadata."
            )

        idx = pd.to_datetime(ts, errors="coerce")
        # if naive, assume UTC (your normalized files should be tz-aware already though)
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")

        df.index = idx
        df.index.name = "ts_event"
        yield df


def sell_at_bid_from_trades_parquet_chunked(
    trades_parquet: pathlib.Path,
    bbo_grid: pd.DataFrame,
    *,
    tick: float,
    freq: str = "1s",
    batch_rows: int = 2_000_000,
    tol: Optional[str] = None,
) -> pd.Series:
    """
    Computes sell_at_bid on a freq grid without loading the full trades parquet.

    Logic:
      - stamp each trade with the most recent bid (merge_asof backward) from bbo_grid
      - mark sell prints ('A') at bid (price == bid in ticks)
      - aggregate size to freq bins
    """
    if bbo_grid.empty:
        return pd.Series(dtype="float64", name="sell_at_bid_1s")

    # Use bbo_grid's timezone as canonical
    tz = bbo_grid.index.tz
    if tz is None:
        raise ValueError("bbo_grid index must be tz-aware.")

    # merge tolerance default: one bar width
    tol_td = _timedelta_from_freq(tol) if tol is not None else _timedelta_from_freq(freq)

    cols = ["price", "size", "side"]
    acc: Dict[pd.Timestamp, float] = {}

    prev_tail: Optional[pd.DataFrame] = None

    for chunk in _parquet_batches(trades_parquet, columns=cols, batch_rows=batch_rows):
        # chunk index currently UTC; convert to bbo tz to align merge_asof
        # (if it already matches, tz_convert is still safe)
        if chunk.index.tz is None:
            chunk.index = chunk.index.tz_localize("UTC")
        chunk = chunk.tz_convert(tz).sort_index()

        if prev_tail is not None and not prev_tail.empty:
            chunk = pd.concat([prev_tail, chunk], axis=0).sort_index()

        # keep small overlap for merge boundary safety
        prev_tail = chunk.iloc[-5000:].copy()

        stamped = pd.merge_asof(
            chunk,
            bbo_grid[["bid_px_00"]].sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=tol_td,
        )

        x = stamped.dropna(subset=["price", "size", "side", "bid_px_00"])
        if x.empty:
            continue

        x["price"] = pd.to_numeric(x["price"], errors="coerce")
        x["bid_px_00"] = pd.to_numeric(x["bid_px_00"], errors="coerce")
        x["size"] = pd.to_numeric(x["size"], errors="coerce")
        x = x.dropna(subset=["price", "bid_px_00", "size"])
        if x.empty:
            continue

        px_ticks = (x["price"] / tick).round().astype("Int64")
        bid_ticks = (x["bid_px_00"] / tick).round().astype("Int64")

        is_sell = x["side"].astype("string").eq("A")
        at_bid = px_ticks.eq(bid_ticks)
        m = is_sell & at_bid
        if not m.any():
            continue

        g = x.index.floor(freq)
        s = x.loc[m, "size"].groupby(g[m]).sum()

        for ts, v in s.items():
            acc[ts] = acc.get(ts, 0.0) + float(v)

    out = pd.Series(acc, dtype="float64")
    out = out.sort_index()
    out.name = "sell_at_bid_1s"
    return out


# ============================================================
# 4) JOIN to df0 + labeling + mid diagnostics
# ============================================================

def build_df0_grid(tradebars: pd.DataFrame, bbo_grid: pd.DataFrame) -> pd.DataFrame:
    df0 = tradebars.join(bbo_grid, how="left")
    df0 = df0.dropna(subset=["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
    return df0


def attach_sell_at_bid(df0: pd.DataFrame, sell_at_bid: pd.Series) -> pd.DataFrame:
    out = df0.copy()
    out["sell_at_bid_1s"] = sell_at_bid.reindex(out.index).fillna(0.0)
    return out


def label_passive_bid_fill(df: pd.DataFrame, *, H: int, my_size: int, join_frac: float, rth_end: str) -> pd.DataFrame:
    """
    Fill if sum_{t+1..t+H} sell_at_bid >= bid_sz_00(t)*join_frac + my_size
    Session-safe via groupby(session_date).
    """
    x = df.copy()
    x["sell_at_bid_1s"] = pd.to_numeric(x["sell_at_bid_1s"], errors="coerce").fillna(0.0).astype("float64")
    bid_sz = pd.to_numeric(x["bid_sz_00"], errors="coerce").fillna(0.0).astype("float64")
    x["queue_ahead"] = bid_sz * float(join_frac)

    if "session_date" not in x.columns:
        x["session_date"] = x.index.normalize()

    def one_day(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        day = d.index[0].normalize()
        close_ts = day + pd.to_timedelta(rth_end)
        ttc = (close_ts - d.index).total_seconds()
        d["can_quote"] = (ttc >= float(H))

        sell = d["sell_at_bid_1s"]
        d["consumption_nextH"] = sell.rolling(H, min_periods=H).sum().shift(-H)

        needed = d["queue_ahead"] + float(my_size)
        d["fill_bid_H"] = ((d["consumption_nextH"] >= needed).fillna(False) & d["can_quote"]).astype("int8")
        return d

    return x.groupby("session_date", group_keys=False).apply(one_day)


def add_mid_and_forward_moves(df: pd.DataFrame, *, horizons: Iterable[int], rth_end: str) -> pd.DataFrame:
    x = df.copy()
    x["bid_px_00"] = pd.to_numeric(x["bid_px_00"], errors="coerce")
    x["ask_px_00"] = pd.to_numeric(x["ask_px_00"], errors="coerce")

    x["spread"] = (x["ask_px_00"] - x["bid_px_00"]).astype("float64")
    x["mid"] = 0.5 * (x["bid_px_00"] + x["ask_px_00"])

    if "session_date" not in x.columns:
        x["session_date"] = x.index.normalize()

    def one_day(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        day = d.index[0].normalize()
        close_ts = day + pd.to_timedelta(rth_end)
        ttc = (close_ts - d.index).total_seconds()

        sp = d["spread"].replace(0.0, np.nan)
        for k in horizons:
            d[f"can_eval_{k}s"] = (ttc >= float(k))
            d[f"mid_fwd_{k}s"] = d["mid"].shift(-k)
            d[f"mid_move_{k}s"] = d[f"mid_fwd_{k}s"] - d["mid"]
            d[f"mid_move_{k}s_spread"] = d[f"mid_move_{k}s"] / sp
        return d

    return x.groupby("session_date", group_keys=False).apply(one_day)


def summarize_mid_move_conditional(df: pd.DataFrame, *, k: int, use_spread_units: bool) -> pd.DataFrame:
    col = f"mid_move_{k}s_spread" if use_spread_units else f"mid_move_{k}s"
    can = f"can_eval_{k}s"

    m = df.loc[df[can] & df["mid"].notna(), ["fill_bid_H", col]].dropna()

    out = {}
    for label, name in [(1, "FILLED"), (0, "NOT_FILLED")]:
        v = m.loc[m["fill_bid_H"] == label, col]
        out[name] = {
            "count": int(len(v)),
            "mean": float(v.mean()) if len(v) else np.nan,
            "median": float(v.median()) if len(v) else np.nan,
            "p10": float(v.quantile(0.10)) if len(v) else np.nan,
            "p50": float(v.quantile(0.50)) if len(v) else np.nan,
            "p90": float(v.quantile(0.90)) if len(v) else np.nan,
        }
    return pd.DataFrame(out)
