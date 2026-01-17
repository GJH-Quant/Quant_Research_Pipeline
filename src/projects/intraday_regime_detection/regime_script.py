# =======================================
# INTRADAY REGIME CLASSIFICATION (HMM)
# =======================================

import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import databento as db

# =========================
# CONFIG (EDIT THESE)
# =========================
SYMBOL = "ABNB"
START_DATE = "2025-01-01"
TIME_INT = "60s"
TRAIN_FRAC = 0.7
N_STATES = 3
RTH_START = "09:30:00"

# Data paths
IN_PATH = pathlib.Path(r"Directory")
OUT_PATH = pathlib.Path(r"Directory")

# Run flags
BUILD_PARQUETS = False          # set True only when needed
OVERWRITE_PARQUETS = False      # set True only if you want to rebuild


# =========================
# RAW TRADES -> PARQUET
# =========================
def trades_to_parquet(
        in_path: pathlib.Path,
        out_path: pathlib.Path,
):

    paths = sorted(in_path.glob("*dbn.zst"))

    for path in paths:
        print('Loading:', path.name)
        store = db.DBNStore.from_file(path)
        df = store.to_df()

        df = df[[
            'ts_event',
            'side',
            'price',
            'size',
            'sequence',
            'symbol'
        ]]

        df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)

        df = df.set_index('ts_event')
        df = df.tz_convert('America/New_York')
        df = df.between_time('09:30','16:00', inclusive='left')

        df['side'] = (
            df['side']
            .astype('string')
            .str.strip()
            .str.upper()
            .astype('category')
        )

        df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('int64', copy=False)

        df['size'] = pd.to_numeric(df['size'], errors='coerce').astype('UInt32')
        df['sequence'] = pd.to_numeric(df['sequence'], errors='coerce').astype('UInt32')

        df = df.loc[(df['price'] > 0) & (df['size'] > 0)]

        df = df.reset_index().sort_values(['ts_event', 'sequence']).set_index('ts_event')

        x = path.name.removesuffix(".trades.dbn.zst").removeprefix("xnas-itch-")
        out = out_path / f'{x}.trades.parquet'

        df.to_parquet(out)

    print(f"[OK] Parquets ready at: {out_path}")


# =========================
# PARQUET -> DF
# =========================
def parquet_to_df(in_path: pathlib.Path):

    paths = sorted(in_path.glob("*.parquet"))
    dfs = []

    for path in paths: 

        df = pd.read_parquet(path)
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0)
    
    return dfs


# =========================
# TIME BARS
# =========================
def create_time_bars(
    df: pd.DataFrame,
    time_int: str,
    bool_fill: bool,
    rth_start: str = "09:30:00",
    price_col: str = "price",
    size_col: str = "size",
):
    
    x = df[[price_col, size_col]].copy()
    x[price_col] = x[price_col].astype("float64")

    session_key = x.index.normalize()

    def one_session(data: pd.DataFrame) -> pd.DataFrame:
        rth_anchor = data.index[0].normalize() + pd.to_timedelta(rth_start)

        bars = data.resample(
            time_int,
            origin=rth_anchor,
            label="left",
            closed="left",
        ).agg(
            open=(price_col, "first"),
            high=(price_col, "max"),
            low=(price_col, "min"),
            close=(price_col, "last"),
            volume=(size_col, "sum"),
            n_trades=(price_col, "count"),
        ).asfreq(time_int)

        bars["dollar"] = (data[price_col] * data[size_col]).resample(
            time_int,
            origin=rth_anchor,
            label="left",
            closed="left",
        ).sum().asfreq(time_int)

        bars["session_date"] = bars.index.normalize().date

        if bool_fill:
            bars["close"] = bars["close"].ffill()
            bars["volume"] = bars["volume"].fillna(0)
            bars["n_trades"] = bars["n_trades"].fillna(0).astype("int64")
            bars["dollar"] = bars["dollar"].fillna(0)

            bars = bars.dropna(subset=["close"])
            bars["inactive"] = bars["n_trades"] == 0

            c = bars.loc[bars["inactive"], "close"]
            bars.loc[bars["inactive"], ["open", "high", "low"]] = np.column_stack([c, c, c])
            bars[["open", "high", "low"]] = bars[["open", "high", "low"]].fillna(bars["close"])
        else:
            bars = bars.dropna(subset=["close"])
            bars["inactive"] = False

        return bars

    bars = x.groupby(session_key, group_keys=False).apply(one_session)

    # --- sanity prints (keep these) ---
    bars_per_day = bars.groupby(bars.index.normalize()).size()
    label = "filled" if bool_fill else "unfilled"
    print(f"[SANITY] bars/day summary ({label}):")
    print(bars_per_day.describe())

    rth_start_t = pd.to_timedelta(rth_start)
    rth_end_t = pd.to_timedelta("16:00:00")
    bar_times = bars.index - bars.index.normalize()
    outside_rth = (bar_times < rth_start_t) | (bar_times >= rth_end_t)
    n_outside = int(outside_rth.sum())

    if n_outside:
        print(f"[SANITY] WARNING: {n_outside} bars outside RTH")
        print(bars.loc[outside_rth].head(5))
    else:
        print("[SANITY] OK: no bars outside RTH")

    print(f"[SANITY] inactive rate: {bars['inactive'].mean():.2%}")
    frac_h_eq_l = float((bars["high"] == bars["low"]).mean())
    print(f"[SANITY] fraction(high==low): {frac_h_eq_l:.2%}")

    return bars


# =========================
# FEATURES
# =========================
def add_log_hl_vol(
    bars: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
):
    
    
    out = bars.copy()
    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")

    out["log_hl"] = np.log(h / l)
    out["log_hl"] = out["log_hl"].replace([np.inf, -np.inf], np.nan)
    
    return out


def tod_normalization(
    bars: pd.DataFrame,
    col: str = "volume",
    tod_int: str = "5min",
    rth_start: str = "09:30:00",
):

    out = bars.copy()

    rth_open = out.index.normalize() + pd.to_timedelta(rth_start)
    seconds_since_open = (out.index - rth_open).total_seconds()
    tod_step = pd.to_timedelta(tod_int).total_seconds()
    out["tod_bin"] = (seconds_since_open // tod_step).astype("int64")

    stats = out.groupby("tod_bin")[col].agg(
        tod_median="median",
        tod_mad=lambda x: np.median(np.abs(x - np.median(x))),
    )

    out = out.join(stats, on="tod_bin")
    out[f"{col}_tod_z"] = (out[col] - out["tod_median"]) / out["tod_mad"].replace(0, np.nan)

    return out.drop(columns=["tod_bin", "tod_median", "tod_mad"])


# =========================
# TRAIN/TEST SPLIT (by session)
# =========================
def train_test_split(
    bars: pd.DataFrame,
    train_frac: float = 0.7,
):
    
    sessions = (
        bars[["session_date"]]
        .drop_duplicates()
        .sort_values("session_date")
        .reset_index(drop=True)
    )

    split = int(len(sessions) * train_frac)
    train_sessions = sessions.iloc[:split, 0]
    test_sessions = sessions.iloc[split:, 0]

    train_df = bars[bars["session_date"].isin(train_sessions)].copy()
    test_df = bars[bars["session_date"].isin(test_sessions)].copy()

    return train_df, test_df


# =========================
# HMM FIT
# =========================
def fit_hmm_intraday_time_bars(
    bars: pd.DataFrame,
    col: str,
    states: int,
    train_frac: float = 0.7,
    lag_for_trading: int = 1,
    state_col: str = "states",
):
    
    out = bars.copy()
  
    good_idx = out[col].dropna().index
    model_df = out.loc[good_idx].sort_index()

    train_df, test_df = train_test_split(model_df, train_frac=train_frac)
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    X_train = train_df[[col]].to_numpy()
    X_test = test_df[[col]].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_lengths = train_df.groupby("session_date", sort=True).size().to_list()
    test_lengths = test_df.groupby("session_date", sort=True).size().to_list()

    model = GaussianHMM(
        n_components=states,
        covariance_type="diag",
        n_iter=300,
        random_state=42,
    )

    model.fit(X_train, lengths=train_lengths)

    train_states = model.predict(X_train, lengths=train_lengths)
    test_states = model.predict(X_test, lengths=test_lengths)

    out[state_col] = -1
    out.loc[train_df.index, state_col] = train_states
    out.loc[test_df.index, state_col] = test_states
    out[state_col] = out[state_col].astype("int64")

    lag_col = f"{state_col}_lag{lag_for_trading}"
    out[lag_col] = out.groupby("session_date", sort=False)[state_col].shift(lag_for_trading)

    return out


# =========================
# PRINTS (REPORTING)
# =========================
def print_state_summary(
    bars: pd.DataFrame,
    state_col: str = "states",
    feature_col: str | None = None,
    title: str = "",
):
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)

    props = bars[state_col].value_counts(normalize=True).sort_index()
    print("\nState proportions:")
    print(props.to_frame("proportion").round(4))

    stats = {
        "inactive_rate": bars.groupby(state_col)["inactive"].mean(),
        "avg_volume": bars.groupby(state_col)["volume"].mean(),
    }
    if feature_col is not None:
        stats[f"mean_{feature_col}"] = bars.groupby(state_col)[feature_col].mean()

    summary = pd.concat(stats, axis=1).round(6)
    print("\nState characteristics:")
    print(summary)


def print_transition_matrix(
    bars: pd.DataFrame,
    state_col: str = "states",
    lag_col: str = "states_lag1",
    title: str = "",
):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

    trans = (
        bars.dropna(subset=[lag_col])
        .groupby([lag_col, state_col])
        .size()
        .unstack(fill_value=0)
    )
    trans = trans.div(trans.sum(axis=1), axis=0)
    trans.index.name = "from"
    trans.columns.name = "to"
    print(trans.round(4))


# =========================
# MAIN
# =========================

def main():
    
    if BUILD_PARQUETS:
        trades_to_parquet(IN_PATH, OUT_PATH, overwrite=OVERWRITE_PARQUETS)

    df = parquet_to_df(OUT_PATH)
    df = df.loc[START_DATE:].copy()

    bars = create_time_bars(df, time_int=TIME_INT, bool_fill=True, rth_start=RTH_START)

    bars1 = add_log_hl_vol(bars)
    bars2 = tod_normalization(bars, "volume", rth_start=RTH_START)

    bars1_regime = fit_hmm_intraday_time_bars(
        bars1, col="log_hl", states=N_STATES, train_frac=TRAIN_FRAC
    )
    bars2_regime = fit_hmm_intraday_time_bars(
        bars2, col="volume_tod_z", states=N_STATES, train_frac=TRAIN_FRAC
    )

    print_state_summary(
        bars1_regime,
        state_col="states",
        feature_col="log_hl",
        title="Log(H/L) Volatility Regime",
    )
    print_transition_matrix(
        bars1_regime,
        state_col="states",
        lag_col="states_lag1",
        title="Log(H/L) Regime Transition Matrix",
    )
    print_state_summary(
        bars2_regime,
        state_col="states",
        feature_col="volume_tod_z",
        title="TOD Normalized Volume Regime",
    )
    print_transition_matrix(
        bars2_regime,
        state_col="states",
        lag_col="states_lag1",
        title="TOD Normalized Volume Regime Transition Matrix",
    )


if __name__ == "__main__":
    main()