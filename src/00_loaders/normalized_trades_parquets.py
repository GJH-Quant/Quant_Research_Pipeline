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

        df = df[['ts_event',
                'action',
                'side',
                'price',
                'size',
                'sequence',
                'symbol']]

        df = df.tz_convert('America/New_York')
        df = df.reset_index()
        df = df.set_index('ts_event')
        df = df.tz_convert('America/New_York')
        df = df.between_time('09:30','16:00')

        x = path.name.removesuffix(".trades.dbn.zst").removeprefix("xnas-itch-")
        out = out_path / f'{x}.trades.parquet'

        df.to_parquet(out)

in_path = pathlib.Path(r"Directory/File")
out_path = pathlib.Path(r"Directory/File")

trades_to_parquet(in_path, out_path)