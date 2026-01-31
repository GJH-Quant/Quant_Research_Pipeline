# config.py 
from __future__ import annotations 
import pathlib

# =========================
# TIME BARS (RTH)
# =========================
TRADES_PATH = pathlib.Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\trades")
DATE_SLICE = ('2025-01-01', '2025-12-20')

TIME_INT    = '60s'
BOOL_FILL   = True

RTH_START = "09:30:00"
RTH_END   = "16:00:00"

PRICE_COL = "price"
SIDE_COL  = "side"
SIZE_COL  = "size"

# =========================
# VOL FEATURES
# =========================
HIGH_COL = "high"
LOW_COL  = "low"
LOG_RET_COL = "log_ret"

RV_EWMA_SPAN   = 5
RV_MIN_PERIODS = 5
RV_OUT_COL     = "rv_ewma"

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

STATE_COL_RV = "states_rv"
STATE_COL_HL = "states_hl"