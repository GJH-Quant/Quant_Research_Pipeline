import pandas as pd
import numpy as np
import pathlib

def parquet_to_df(in_path: pathlib.Path):

    paths = sorted(in_path.glob("*.parquet"))
    dfs = []

    for path in paths: 

        df = pd.read_parquet(path)
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    return dfs