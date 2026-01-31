# ===================================
# 1D GAUSSIAN HMM SUMMARY STATS
# ===================================

import pandas as pd
import numpy as np

def hmm_summary_stats(
    bars: pd.DataFrame,
    feature_col: str,
    states_col: str,
    session_col: str,
    inactive_col: str = "inactive",
    volume_col: str = "volume",
    logret_col: str = "log_ret",
) -> None:

    print("\n==============================")
    print("HMM SUMMARY STATS")
    print("==============================")

    states_count = bars[states_col].value_counts(dropna=True).sort_index()
    print("\nState counts:")
    print(states_count)

    xr = bars[[states_col, session_col]].dropna(subset=[states_col]).copy()
    xr[states_col] = xr[states_col].astype("Int64")

    new_run = (
        xr[states_col].ne(xr[states_col].shift(1))
        | xr[session_col].ne(xr[session_col].shift(1))
    )

    run_id = new_run.cumsum()
    run_len = xr.groupby(run_id).size()
    run_state = xr.groupby(run_id)[states_col].first()

    avg_run_len = run_len.groupby(run_state).mean()

    print("\n------------------------------")
    print("Per-state diagnostics")
    print("------------------------------")

    print("\nInactive rate:")
    print(bars.groupby(states_col)[inactive_col].mean())

    print("\nAverage volume:")
    print(bars.groupby(states_col)[volume_col].mean())

    print("\nStd(log_ret):")
    print(bars.groupby(states_col)[logret_col].std())

    print("\nAverage feature value:")
    print(bars.groupby(states_col)[feature_col].mean())

    print("\nAverage run length (bars):")
    print(avg_run_len)
