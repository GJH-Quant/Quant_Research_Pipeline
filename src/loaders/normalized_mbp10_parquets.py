# =========================================
#  RAW MBP-10 --> NORMALIZED PARQUETS
# =========================================

import pandas as pd
import numpy as np
import pathlib
import databento as db


MBP10_COLS = [
    "ts_event", "sequence",
    "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
    "bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01",
    "bid_px_02", "ask_px_02", "bid_sz_02", "ask_sz_02",
    "bid_px_03", "ask_px_03", "bid_sz_03", "ask_sz_03",
    "bid_px_04", "ask_px_04", "bid_sz_04", "ask_sz_04",
    "bid_px_05", "ask_px_05", "bid_sz_05", "ask_sz_05",
    "bid_px_06", "ask_px_06", "bid_sz_06", "ask_sz_06",
    "bid_px_07", "ask_px_07", "bid_sz_07", "ask_sz_07",
    "bid_px_08", "ask_px_08", "bid_sz_08", "ask_sz_08",
    "bid_px_09", "ask_px_09", "bid_sz_09", "ask_sz_09",
    "symbol",
]

PX_COLS = [c for c in MBP10_COLS if c.startswith(("bid_px_", "ask_px_"))]
SZ_COLS = [c for c in MBP10_COLS if c.startswith(("bid_sz_", "ask_sz_"))]


def mbp10_to_parquet(in_path: pathlib.Path, out_path: pathlib.Path):
    out_path.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_path.glob("*dbn.zst"))

    for path in paths:
        print("Loading:", path.name)
        store = db.DBNStore.from_file(path)
        df = store.to_df()


        if "ts_event" not in df.columns:
            df = df.reset_index()

        df = df[MBP10_COLS].copy()

        df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)

        df = df.set_index("ts_event")
        df = df.tz_convert("America/New_York")
        df = df.between_time("09:30", "16:00", inclusive="left")


        df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce").astype("UInt32")

        for c in PX_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64", copy=False)

        for c in SZ_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("UInt32")

        df = df.loc[
            (df["sequence"] > 0)
            & (df["bid_px_00"] > 0) & (df["ask_px_00"] > 0)
            & (df["bid_sz_00"] > 0) & (df["ask_sz_00"] > 0)
        ]

        df = df.loc[df["bid_px_00"] <= df["ask_px_00"]]

        df = df.reset_index().sort_values(["ts_event", "sequence"]).set_index("ts_event")

        x = path.name.removesuffix(".mbp-10.dbn.zst").removeprefix("xnas-itch-")
        out = out_path / f"{x}.mbp-10.parquet"
        df.to_parquet(out)

    print(f"[OK] Parquets ready at: {out_path}")
