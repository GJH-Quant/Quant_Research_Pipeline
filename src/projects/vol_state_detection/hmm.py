from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from typing import Tuple

# ============================================
# TRAIN / TEST SPLIT
# ============================================

def train_test_split_by_session(
    bars: pd.DataFrame,
    train_frac: float = 0.7,
    session_col: str = "session_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    sessions = (
        bars[[session_col]]
        .drop_duplicates()
        .sort_values(session_col)
        .reset_index(drop=True)
    )

    split = int(len(sessions) * train_frac)
    train_sessions = sessions.iloc[:split, 0]
    test_sessions = sessions.iloc[split:, 0]

    train_df = bars[bars[session_col].isin(train_sessions)].copy()
    test_df = bars[bars[session_col].isin(test_sessions)].copy()

    return train_df, test_df

# ============================================
# FIT HMM
# ============================================

def fit_hmm_intraday(
    bars: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    states: int,
    train_frac: float = 0.7,
    session_col: str = "session_date",
    state_col: str = "states",
    random_state: int = 42,
) -> pd.DataFrame:
    
    out = bars.copy()

    train = train.dropna(subset=[col]).copy()
    test = test.dropna(subset=[col]).copy()

    if train.empty or test.empty:
        raise ValueError("Likely a feature is all-NaN; check feature engineering/min_periods.")

    X_train = train[[col]].to_numpy()
    X_test = test[[col]].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_lengths = train.groupby(session_col, sort=True).size().to_list()
    test_lengths = test.groupby(session_col, sort=True).size().to_list()

    model = GaussianHMM(
        n_components=states,
        covariance_type="diag",
        n_iter=300,
        random_state=random_state,
    )

    model.fit(X_train, lengths=train_lengths)

    train_states = model.predict(X_train, lengths=train_lengths)
    test_states = model.predict(X_test, lengths=test_lengths)

    out[state_col] = np.nan
    out.loc[train.index, state_col] = train_states
    out.loc[test.index, state_col] = test_states
    out[state_col] = out[state_col].astype("Int64")

    return out

# ============================================
# HMM SUMMARY STATS
# ============================================

def hmm_summary_stats(
        bars: pd.DataFrame,
        feature_col: str,
        states_col: str,
):
    
    states_count = bars[states_col].value_counts()
    print('Value Counts of Regime States:')
    print(states_count)

    xr = bars[[states_col, "session_date"]].dropna(subset=[states_col]).copy()
    xr[states_col] = xr[states_col].astype("Int64")

    new_run = (
        xr[states_col].ne(xr[states_col].shift(1)) |
        xr["session_date"].ne(xr["session_date"].shift(1))
    )

    run_id = new_run.cumsum()
    run_len = xr.groupby(run_id).size()
    run_state = xr.groupby(run_id)[states_col].first()

    avg_run_len = run_len.groupby(run_state).mean()

    stats = {
        'inactive_rate': bars.groupby(states_col)['inactive'].mean(),
        'avg_volume': bars.groupby(states_col)['volume'].mean(),
        'avg_std': bars.groupby(states_col)['log_ret'].std(),
        'avg_run_len_bars': avg_run_len
    }

    print('Summary Stats')
    print('-' * 40)
    print(stats)