# run.py
from __future__ import annotations
import pathlib
import numpy as np

from .config import (
    # bars
    TRADES_PATH, DATE_SLICE, TIME_INT, BOOL_FILL, RTH_START, RTH_END, PRICE_COL, SIDE_COL, SIZE_COL,
    # h/l
    HIGH_COL, LOW_COL, LOG_RET_COL, SESSION_COL,
    RV_EWMA_SPAN, RV_MIN_PERIODS, RV_OUT_COL, TRAIN_FRAC, LOG_HL_OUT_COL,
    N_STATES, HMM_COV_TYPE, HMM_N_ITER, HMM_RANDOM_STATE,
    STATE_COL_HL, STATE_COL_RV,
)

from src.loaders.parquets_to_df import parquet_to_df
from src.bars.create_time_bars_w_flow import create_time_bars_w_flow
from src.features.ewma_realized_vol import add_ewma_realized_vol
from src.features.log_high_low_vol import add_log_hl_vol
from src.labels.train_test_split import train_test_split_by_session
from src.regimes.fit_1D_hmm import fit_hmm_intraday_1d
from src.regimes.hmm_summary_stats import hmm_summary_stats

def main(return_df: bool = False):

    trades = parquet_to_df(TRADES_PATH)
    trades = trades.loc[DATE_SLICE[0]:DATE_SLICE[1]]

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

    # -------------------------
    # Build vol features
    # -------------------------
    bars = add_log_hl_vol(
        bars,
        high_col=HIGH_COL,
        low_col=LOW_COL,
    )

    bars = add_ewma_realized_vol(
        bars,
        logret_col=LOG_RET_COL,
        session_col=SESSION_COL,
        span=RV_EWMA_SPAN,
        min_periods=RV_MIN_PERIODS,
        out_col=RV_OUT_COL,
    )

    # -------------------------
    # Train/Test split (session aware)
    # -------------------------
    train, test = train_test_split_by_session(
        bars,
        train_frac=TRAIN_FRAC,
        session_col=SESSION_COL,
    )

    # -------------------------
    # HMM on RV
    # -------------------------
    print("\n" + "=" * 60)
    print(f"HMM REGIMES ON: {RV_OUT_COL}")
    print("=" * 60)

    df_rv, rv_model, rv_scaler = fit_hmm_intraday_1d(
        bars,
        train,
        test,
        col=RV_OUT_COL,
        states=N_STATES,
        session_col=SESSION_COL,
        state_col=STATE_COL_RV,
        covariance_type=HMM_COV_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE,
    )

    hmm_summary_stats(df_rv, feature_col=RV_OUT_COL, states_col=STATE_COL_RV, session_col=SESSION_COL)

    # -------------------------
    # HMM on log(H/L)
    # -------------------------
    print("\n" + "=" * 60)
    print(f"HMM REGIMES ON: {LOG_HL_OUT_COL}")
    print("=" * 60)

    df_hl, hl_model, hl_scaler = fit_hmm_intraday_1d(
        bars,
        train,
        test,
        col=LOG_HL_OUT_COL,
        states=N_STATES,
        session_col=SESSION_COL,
        state_col=STATE_COL_HL,
        covariance_type=HMM_COV_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE,
    )

    hmm_summary_stats(df_hl, feature_col=LOG_HL_OUT_COL, states_col=STATE_COL_HL, session_col=SESSION_COL)
    
    if return_df:
        # Return ONLY the main features + states, indexed by timestamp
        out = df_rv[[RV_OUT_COL, STATE_COL_RV]].copy()
        out[LOG_HL_OUT_COL] = df_hl[LOG_HL_OUT_COL]
        out[STATE_COL_HL]   = df_hl[STATE_COL_HL]
        return out.sort_index()

if __name__ == "__main__":
    main()