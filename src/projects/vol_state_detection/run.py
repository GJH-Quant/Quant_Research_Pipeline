from __future__ import annotations

import pathlib

from .config import (
    # data
    TRADES_DIR,
    # bars
    TIME_INT, BOOL_FILL, RTH_START, RTH_END, PRICE_COL, SIZE_COL,
    # features
    HIGH_COL, LOW_COL, LOG_HL_OUT_COL,
    RV_EWMA_SPAN, RV_MIN_PERIODS, RV_OUT_COL,
    SESSION_COL,
    # hmm
    TRAIN_FRAC, N_STATES, HMM_COV_TYPE, HMM_N_ITER, HMM_RANDOM_STATE,
    STATE_COL_RV, STATE_COL_HL,
)

from .bars import parquet_to_df, create_time_bars
from .features import build_vol_features
from .hmm import train_test_split_by_session, fit_hmm_intraday, hmm_summary_stats

# =========================
# RUNTIME INPUTS (keep out of config)
# =========================
DATE_SLICE = ("2025-01-01", "2025-12-20")


def main() -> dict:
    # -------------------------
    # Load trades
    # -------------------------
    trades = parquet_to_df(pathlib.Path(TRADES_DIR))
    if DATE_SLICE is not None:
        trades = trades.loc[DATE_SLICE[0]:DATE_SLICE[1]]

    # -------------------------
    # Build RTH time bars
    # -------------------------
    bars = create_time_bars(
        trades,
        time_int=TIME_INT,
        bool_fill=BOOL_FILL,
        rth_start=RTH_START,
        rth_end=RTH_END,
        price_col=PRICE_COL,
        size_col=SIZE_COL,
    )

    # -------------------------
    # Build vol features
    # -------------------------
    bars = build_vol_features(
        bars,
        high_col=HIGH_COL,
        low_col=LOW_COL,
        log_hl_out_col=LOG_HL_OUT_COL,
        logret_col="log_ret",
        session_col=SESSION_COL,
        rv_span=RV_EWMA_SPAN,
        rv_min_periods=RV_MIN_PERIODS,
        rv_out_col=RV_OUT_COL,
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

    df_rv = fit_hmm_intraday(
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

    df_hl = fit_hmm_intraday(
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

    return {"bars": bars, "rv_states": df_rv, "hl_states": df_hl}


if __name__ == "__main__":
    main()
