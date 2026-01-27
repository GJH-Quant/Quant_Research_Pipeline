import pathlib

# =========================
# CONFIG
# =========================
SYMBOL = "AAPL"

TIME_INT   = "60s"
TRAIN_FRAC = 0.7
N_STATES   = 3

RTH_START = "09:30:00"
RTH_END   = "16:00:00"
TZ        = "America/New_York"

# Data paths
IN_PATH = pathlib.Path(r"Directory")
OUT_PATH = pathlib.Path(r"Directory")

# Run flags
BUILD_PARQUETS     = False
OVERWRITE_PARQUETS = False

# Feature params
RV_EWMA_SPAN    = 5
RV_MIN_PERIODS  = 5