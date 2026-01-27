# config.py
from __future__ import annotations

# =========================
# TIME BARS (RTH)
# =========================
TIME_INT = "60s"
BOOL_FILL = True

RTH_START = "09:30:00"
RTH_END   = "16:00:00"

PRICE_COL = "price"
SIZE_COL  = "size"

# =========================
# BURST FEATURES
# =========================
BURST_FEATURE_COL = "volume"   # in bars after resample (volume, dollar, n_trades, etc.)
TOD_COL = "tod_bin"

# Rolling percentile burst
PCT_LOOKBACK = 60
PCT_MIN_PERIODS = 20
PCT_Q = 0.995
PCT_EPS = 1e-12
PCT_OUT_COL = "burst_pct_score"

# Median/MAD burst
MAD_LOOKBACK = 60
MAD_MIN_PERIODS = 20
MAD_FLOOR = 1e-6
MAD_SCALE = 1.4826
MAD_OUT_COL = "burst_mad_z"

# =========================
# SANITY (optional thresholds)
# =========================
PCT_APPROX_EVENT_SCORE = 1.0   # event ~ score>1 => log1p(score)>log1p(1)
MAD_EVENT_Z = 3.0              # typical robust z threshold