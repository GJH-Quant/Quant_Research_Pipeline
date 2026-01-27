import pathlib

from .config import (
    OUT_PATH,
    TIME_INT,
    TRAIN_FRAC,
    N_STATES,
    RV_EWMA_SPAN,
    RV_MIN_PERIODS,
)

from src.projects.vol_state_detection.bars import create_time_bars, parquet_to_df
from src.projects.vol_state_detection.features import add_log_hl_vol, add_ewma_realized_vol
from src.projects.vol_state_detection.hmm import fit_hmm_intraday, train_test_split_by_session, hmm_summary_stats

def main():
    trades_path = pathlib.Path(OUT_PATH)  # already a Path, but safe

    trades = parquet_to_df(trades_path)
    trades = trades.loc["2025-01-01":"2025-12-20"]

    bars = create_time_bars(trades, time_int=TIME_INT, bool_fill=True)

    bars = add_log_hl_vol(bars)
    bars = add_ewma_realized_vol(bars, span=RV_EWMA_SPAN, min_periods=RV_MIN_PERIODS)

    train, test = train_test_split_by_session(bars, train_frac=TRAIN_FRAC)

    print("\n" + "=" * 60)
    print("HMM REGIMES ON: rv_ewma")
    print("=" * 60)

    df_rv = fit_hmm_intraday(
        bars, train, test,
        col="rv_ewma",
        states=N_STATES,
        state_col="states_rv",
    )
    hmm_summary_stats(df_rv, feature_col="rv_ewma", states_col="states_rv")

    print("\n" + "=" * 60)
    print("HMM REGIMES ON: log_hl")
    print("=" * 60)

    df_hl = fit_hmm_intraday(
        bars, train, test,
        col="log_hl",
        states=N_STATES,
        state_col="states_hl",
    )
    hmm_summary_stats(df_hl, feature_col="log_hl", states_col="states_hl")

    return {"bars": bars, "rv_states": df_rv, "hl_states": df_hl}


if __name__ == "__main__":
    main()