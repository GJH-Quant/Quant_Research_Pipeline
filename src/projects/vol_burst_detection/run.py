# run.py
from __future__ import annotations
import pathlib
import numpy as np

from .config import (
    # bars
    TRADES_PATH, DATE_SLICE, TIME_INT, BOOL_FILL, RTH_START, RTH_END, PRICE_COL, SIDE_COL, SIZE_COL,
    # tod binning
    TOD_INT,
    # burst
    PCT_LOOKBACK, PCT_EPS, MAD_LOOKBACK, MAD_FLOOR, MAD_SCALE, VERBOSE
)

from src.loaders.parquets_to_df import parquet_to_df
from src.bars.create_time_bars_w_flow import create_time_bars_w_flow
from src.features.med_mad_tod_norm import add_med_mad_norm
from src.features.percentile_burst import add_rolling_percentile_burst

def main():

    trades = parquet_to_df(TRADES_PATH)
    trades = trades.loc[DATE_SLICE[0]: DATE_SLICE[1]]
    
    # -------------------------
    # Build RTH time bars
    # -------------------------
    bars = create_time_bars_w_flow(
        trades,
        time_int=TIME_INT,
        bool_fill=BOOL_FILL,
        rth_start=RTH_START,
        rth_end=RTH_END,
        price_col=PRICE_COL,
        side_col=SIDE_COL,
        size_col=SIZE_COL,
    )

    bars['log_volume'] = np.log1p(bars['volume'])

    # -------------------------
    # MED MAD NORM
    # -------------------------
    bars = add_med_mad_norm(
        bars,
        feature_col='log_volume',
        bucket_td=TOD_INT,
        time_int_td=TIME_INT,
        rth_start=RTH_START,
        rth_end=RTH_END,
        lookback_days=MAD_LOOKBACK,
        mad_floor=MAD_FLOOR,
        scale_mad=MAD_SCALE,
        verbose=VERBOSE,
    )

    # -------------------------
    # PERCENTILE NORM
    # -------------------------
    bars = add_rolling_percentile_burst(
        bars,
        feature_col='log_volume',
        bucket_td=TOD_INT,
        time_int_td=TIME_INT,
        rth_start=RTH_START,
        rth_end=RTH_END,
        lookback_days=PCT_LOOKBACK,
        eps=PCT_EPS,
        verbose=VERBOSE,
    )    

if __name__ == '__main__':
    main()