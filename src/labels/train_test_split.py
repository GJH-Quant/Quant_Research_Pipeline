# ==============================
# TRAIN/TEST SPLIT
# ==============================

import pandas as pd
import numpy as np

def train_test_split_by_session(
    bars: pd.DataFrame,
    train_frac: float,
    session_col: str,
):
    
    sessions = (
        bars[[session_col]]
        .drop_duplicates()
        .sort_values(session_col)
        .reset_index(drop=True)
    )

    split = int(len(sessions) * train_frac)
    train_sessions = sessions[session_col].iloc[:split]
    test_sessions  = sessions[session_col].iloc[split:]

    train_df = bars[bars[session_col].isin(train_sessions)].copy()
    test_df = bars[bars[session_col].isin(test_sessions)].copy()
    
    return train_df, test_df