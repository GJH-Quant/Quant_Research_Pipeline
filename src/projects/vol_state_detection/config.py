from __future__ import annotations

import pathlib

# =========================
# DATA
# =========================
SYMBOL = "AAPL"
TRADES_DIR = pathlib.Path(r"Directory")

# =========================
# TIME BARS (RTH)
# =========================
TIME_INT = "60s"
BOOL_FILL = True

RTH_START = "09:30:00"
RTH_END   = "16:00:00"
TZ        = "America/New_York"

PRICE_COL = "price"
SIZE_COL  = "size"

# =========================
# VOL FEATURES
# =========================
HIGH_COL = "high"
LOW_COL  = "low"
CLOSE_COL = "close"

# EWMA RV params (within-session)
RV_EWMA_SPAN   = 5
RV_MIN_PERIODS = 5
RV_OUT_COL     = "rv_ewma"

# Range vol proxy
LOG_HL_OUT_COL = "log_hl"

# =========================
# HMM
# =========================
TRAIN_FRAC = 0.7
SESSION_COL = "session_date"

N_STATES = 3
HMM_COV_TYPE = "diag"
HMM_N_ITER = 300
HMM_RANDOM_STATE = 42

# Output columns (two models)
STATE_COL_RV = "states_rv"
STATE_COL_HL = "states_hl"
