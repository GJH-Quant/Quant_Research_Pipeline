# =========================
# FILE: modeling.py
# =========================
from __future__ import annotations

from typing import Iterable, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .ev import add_forward_quotes_H


def ensure_session_date(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "session_date" not in x.columns:
        x["session_date"] = x.index.normalize()
    return x


def time_split_days(index: pd.DatetimeIndex, train_frac: float) -> Tuple[Set[pd.Timestamp], Set[pd.Timestamp]]:
    days = sorted(pd.Index(index.normalize().unique()))
    cut = int(len(days) * float(train_frac))
    return set(days[:cut]), set(days[cut:])


def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Past-only, session-safe features. Rolling stats shifted by 1.
    """
    x = ensure_session_date(df).copy()

    for c in ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "net_aggr_frac", "aggr_frac"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    x["spread_t"] = (x["ask_px_00"] - x["bid_px_00"]).astype("float64")
    x["mid_t"] = 0.5 * (x["bid_px_00"] + x["ask_px_00"])

    denom = (x["bid_sz_00"] + x["ask_sz_00"]).replace(0.0, np.nan)
    x["imb_t"] = ((x["bid_sz_00"] - x["ask_sz_00"]) / denom).astype("float64").fillna(0.0)

    x["mid_ret_1s"] = x.groupby("session_date", group_keys=False)["mid_t"].apply(lambda s: np.log(s).diff())
    x["mid_ret_1s"] = x["mid_ret_1s"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lags = (1, 2, 3, 5, 10)
    wins = (10, 30, 60)

    base_cols = ["mid_ret_1s", "imb_t", "spread_t"]
    if "net_aggr_frac" in x.columns:
        x["net_aggr_frac"] = x["net_aggr_frac"].fillna(0.0)
        base_cols.append("net_aggr_frac")

    for k in lags:
        for c in base_cols:
            x[f"{c}_lag{k}"] = x.groupby("session_date", group_keys=False)[c].shift(k)

    for w in wins:
        x[f"rv_roll{w}"] = x.groupby("session_date", group_keys=False)["mid_ret_1s"].apply(
            lambda s: s.rolling(w, min_periods=w).std(ddof=0).shift(1)
        )
        x[f"imb_roll{w}_std"] = x.groupby("session_date", group_keys=False)["imb_t"].apply(
            lambda s: s.rolling(w, min_periods=w).std(ddof=0).shift(1)
        )
        x[f"spread_roll{w}_mean"] = x.groupby("session_date", group_keys=False)["spread_t"].apply(
            lambda s: s.rolling(w, min_periods=w).mean().shift(1)
        )

    lag_suffixes = tuple(f"_lag{k}" for k in lags)
    feature_cols = [c for c in x.columns if c.endswith(lag_suffixes)] + [
        "spread_t", "imb_t", "mid_ret_1s",
        "rv_roll10", "rv_roll30", "rv_roll60",
        "imb_roll10_std", "imb_roll30_std", "imb_roll60_std",
        "spread_roll10_mean", "spread_roll30_mean", "spread_roll60_mean",
    ]
    feature_cols = [c for c in feature_cols if c in x.columns]

    return x[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_toxic_label(df: pd.DataFrame, H: int, spread_mult: float) -> pd.Series:
    x = add_forward_quotes_H(df, H=H)
    markout = x["mid_tH"] - x["bid_px_00"]
    y = (markout <= (-float(spread_mult) * x["spread_t"].replace(0.0, np.nan))).astype("int8")
    y.name = "toxic"
    return y


def fit_xgb_classifier(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_te: pd.Series) -> XGBClassifier:
    pos = float(y_tr.mean())
    spw = (1.0 - pos) / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=8,
        scale_pos_weight=spw,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model


def run_xgb_score_gating(
    df0: pd.DataFrame,
    H: int,
    my_size: int,
    join_frac: float,
    toxic_spread_mult: float,
    score_mins: Iterable[float],
    train_frac_days: float,
    rth_end: str,
    *,
    label_passive_bid_fill,
) -> pd.DataFrame:
    """
    Returns the gating sweep table.
    label_passive_bid_fill injected to avoid importing labeling.py here.
    """
    df = label_passive_bid_fill(df0, H=H, my_size=my_size, join_frac=join_frac, rth_end=rth_end)
    df = add_forward_quotes_H(df, H=H)

    valid = (
        df["can_quote"].astype(bool)
        & df["can_eval_H"].astype(bool)
        & df["bid_px_00"].notna()
        & df["spread_t"].notna()
    )
    score_idx = df.index[valid]
    if len(score_idx) < 100_000:
        raise RuntimeError(f"Too few valid quotes: {len(score_idx):,}")

    X = build_xgb_features(df)
    y_fill = df.loc[score_idx, "fill_bid_H"].astype(int)
    y_toxic = make_toxic_label(df, H=H, spread_mult=toxic_spread_mult).astype(int)

    tr_days, te_days = time_split_days(pd.DatetimeIndex(score_idx), train_frac=train_frac_days)
    tr_idx = pd.Index(score_idx)[pd.Index(score_idx).normalize().isin(tr_days)]
    te_idx = pd.Index(score_idx)[pd.Index(score_idx).normalize().isin(te_days)]

    fill_model = fit_xgb_classifier(X.loc[tr_idx], y_fill.loc[tr_idx], X.loc[te_idx], y_fill.loc[te_idx])
    p_fill = pd.Series(fill_model.predict_proba(X.loc[score_idx])[:, 1], index=score_idx, name="p_fill")

    filled = df.loc[score_idx, "fill_bid_H"].astype(np.int8) == 1
    tox_idx = pd.Index(score_idx)[filled.to_numpy()]
    if len(tox_idx) < 50_000:
        raise RuntimeError(f"Too few filled samples to train toxicity: {len(tox_idx):,}")

    tr_days_t, te_days_t = time_split_days(pd.DatetimeIndex(tox_idx), train_frac=train_frac_days)
    tr_idx_t = tox_idx[tox_idx.normalize().isin(tr_days_t)]
    te_idx_t = tox_idx[tox_idx.normalize().isin(te_days_t)]

    tox_model = fit_xgb_classifier(X.loc[tr_idx_t], y_toxic.loc[tr_idx_t], X.loc[te_idx_t], y_toxic.loc[te_idx_t])
    p_toxic = pd.Series(tox_model.predict_proba(X.loc[score_idx])[:, 1], index=score_idx, name="p_toxic")

    score = (p_fill * (1.0 - p_toxic)).rename("score")

    rows = []
    for smin in score_mins:
        q_idx = score.index[score > float(smin)]
        if len(q_idx) == 0:
            continue

        f_q = df.loc[q_idx, "fill_bid_H"].astype(float)
        e_q = df.loc[q_idx, "bid_px_00"].astype(float)
        midH_q = df.loc[q_idx, "mid_tH"].astype(float)
        bidH_q = df.loc[q_idx, "bid_tH"].astype(float)

        ev_mid = float((f_q * (midH_q - e_q)).mean())
        ev_bid = float((f_q * (bidH_q - e_q)).mean())

        quote_rate = float(len(q_idx) / len(score_idx))
        rows.append({
            "score_min": float(smin),
            "quote_rate": quote_rate,
            "fill_rate": float(f_q.mean()),
            "ev_per_quote_mid_$": ev_mid,
            "ev_per_quote_bid_$": ev_bid,
            "ev_density_mid_$": quote_rate * ev_mid,
            "ev_density_bid_$": quote_rate * ev_bid,
            "avg_p_fill": float(p_fill.loc[q_idx].mean()),
            "avg_p_toxic": float(p_toxic.loc[q_idx].mean()),
            "avg_score": float(score.loc[q_idx].mean()),
        })

    out = pd.DataFrame(rows).sort_values("score_min").reset_index(drop=True)

    # quick metrics on held-out days
    p_te = fill_model.predict_proba(X.loc[te_idx])[:, 1]
    out.attrs["fill_auc"] = float(roc_auc_score(y_fill.loc[te_idx], p_te))
    out.attrs["fill_ap"] = float(average_precision_score(y_fill.loc[te_idx], p_te))

    p_te_t = tox_model.predict_proba(X.loc[te_idx_t])[:, 1]
    out.attrs["tox_auc"] = float(roc_auc_score(y_toxic.loc[te_idx_t], p_te_t))
    out.attrs["tox_ap"] = float(average_precision_score(y_toxic.loc[te_idx_t], p_te_t))

    return out
