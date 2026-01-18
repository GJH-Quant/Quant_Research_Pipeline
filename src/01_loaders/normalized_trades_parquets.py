# =========================================
#  RAW TRADES --> NORMALIZED PARQUETS
# =========================================

import pandas as pd
import numpy as np
import pathlib
import databento as db

def trades_to_parquet(in_path: pathlib.Path, out_path: pathlib.Path):

    paths = sorted(in_path.glob("*dbn.zst"))

    for path in paths:
        print('Loading:', path.name)
        store = db.DBNStore.from_file(path)
        df = store.to_df()

        df = df[[
            'ts_event',
            'side',
            'price',
            'size',
            'sequence',
            'symbol'
        ]]

        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)

        df = df.set_index('ts_event')
        df = df.tz_convert('America/New_York')
        df = df.between_time('09:30','16:00', inclusive='left')

        df['side'] = (
            df['side']
            .astype('string')
            .str.strip()
            .str.upper()
            .astype('category')
        )

        df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('float64')

        df['size'] = pd.to_numeric(df['size'], errors='coerce').astype('UInt32')
        df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce').astype('UInt32')

        df = df.loc[(df['price'] > 0) & (df['size'] > 0)]

        df = df.reset_index().sort_values(['ts_event', 'sequence']).set_index('ts_event')

        x = path.name.removesuffix(".trades.dbn.zst").removeprefix("xnas-itch-")
        out = out_path / f'{x}.trades.parquet'

        df.to_parquet(out)

    print(f"[OK] Parquets ready at: {out_path}")