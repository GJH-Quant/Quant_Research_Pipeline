# ====================================
# FIT BBO XGBOOST CLASSIFIER
# ====================================

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from src.labels.train_test_split import train_test_split_by_session
import pandas as pd
import numpy as np

def fit_baseline_bbo_xgb(
    bars: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    train_frac: float = 0.7,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    n_jobs: int = -1,
    tree_method: str = "hist",
):
    train, test = train_test_split_by_session(
        bars, train_frac=train_frac, session_col="session_date"
    )

    X_train = train[feature_cols].astype("float32")
    X_test  = test[feature_cols].astype("float32")

    y_train = train[label_col].astype("int8").to_numpy()
    y_test  = test[label_col].astype("int8").to_numpy()

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    spw = neg / max(pos, 1.0)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )

    model.fit(X_train, y_train)

    p_test = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=test.index,
        name=f"p_{label_col}",
    )

    metrics = {
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "pos_rate_train": float(np.mean(y_train)) if len(y_train) else np.nan,
        "pos_rate_test": float(np.mean(y_test)) if len(y_test) else np.nan,
        "scale_pos_weight": float(spw),
    }

    if len(np.unique(y_test)) > 1:
        metrics.update(
            {
                "auc": float(roc_auc_score(y_test, p_test.values)),
                "ap": float(average_precision_score(y_test, p_test.values)),
                "logloss": float(log_loss(y_test, p_test.values, labels=[0, 1])),
                "brier": float(brier_score_loss(y_test, p_test.values)),
            }
        )
    else:
        metrics.update({"auc": np.nan, "ap": np.nan, "logloss": np.nan, "brier": np.nan})

    return model, metrics, p_test