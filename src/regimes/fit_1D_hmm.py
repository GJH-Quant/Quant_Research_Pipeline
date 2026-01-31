# ===================================
# FIT 1D GAUSSIAN HMM
# ===================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

def fit_hmm_intraday_1d(
    bars: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    states: int,
    session_col: str,
    state_col: str,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int = 7,
):

    out = bars.copy()

    train = train.dropna(subset=[col]).copy()
    test  = test.dropna(subset=[col]).copy()

    if train.empty or test.empty:
        raise ValueError("Train or test empty after dropping NaNs.")

    X_train = (
        pd.to_numeric(train[col], errors="coerce")
        .to_numpy(dtype="float64")
        .reshape(-1, 1)
    )
    X_test = (
        pd.to_numeric(test[col], errors="coerce")
        .to_numpy(dtype="float64")
        .reshape(-1, 1)
    )

    if not np.isfinite(X_train).all():
        raise ValueError("Non-finite values in X_train.")
    if not np.isfinite(X_test).all():
        raise ValueError("Non-finite values in X_test.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_lengths = train.groupby(session_col, sort=False).size().to_list()
    test_lengths  = test.groupby(session_col, sort=False).size().to_list()

    model = GaussianHMM(
        n_components=states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )

    model.fit(X_train, lengths=train_lengths)

    train_states = model.predict(X_train, lengths=train_lengths)
    test_states  = model.predict(X_test,  lengths=test_lengths)

    out[state_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[train.index, state_col] = train_states
    out.loc[test.index,  state_col] = test_states
    out[state_col] = out[state_col].astype("Int64")

    return out, model, scaler