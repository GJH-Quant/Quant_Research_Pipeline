# config.py 
from __future__ import annotations 
import pathlib
# ========================= 
# TIME BARS (RTH) 
# ========================= 
TRADES_PATH = pathlib.Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\trades")
DATE_SLICE = ('2025-01-01','2025-12-20') 

TIME_INT    = '2s' 
BOOL_FILL   = True

RTH_START = "09:30:00" 
RTH_END   = "16:00:00"

PRICE_COL = "price" 
SIDE_COL  = "side" 
SIZE_COL  = "size" 

# ========================= 
# TOD BINNING (decoupled) 
# ========================= 
TOD_INT = "60s" 

# ========================= 
# BURST FEATURES 
# ========================= 
PCT_LOOKBACK = 10
PCT_EPS      = 1e-12 

MAD_LOOKBACK = 10 
MAD_FLOOR    = 1e-6 
MAD_SCALE    = 1.4826 

VERBOSE = True