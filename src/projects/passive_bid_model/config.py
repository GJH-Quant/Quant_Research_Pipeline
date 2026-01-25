# =========================
# FILE: config.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class Config:
    # raw inputs
    raw_trades_dir: Path
    raw_mbp1_dir: Path

    # normalized outputs
    norm_trades_dir: Path
    norm_mbp1_dir: Path

    # optional cached grid output (df0 on bar_freq grid)
    grid_dir: Path | None = None

    # market/session params
    tz: str = "America/New_York"
    rth_start: str = "09:30:00"
    rth_end: str = "16:00:00"
    tick: float = 0.01

    # grid / bar params
    bar_freq: str = "1s"  # allow "1s", "5s", etc.

    # model params
    H: int = 5
    my_size: int = 100
    join_fracs: Tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)

    diag_join_frac: float = 1.00
    diag_horizons: Tuple[int, ...] = (1, 5, 10)

    run_xgb: bool = True
    xgb_join_frac: float = 1.00
    toxic_spread_mult: float = 0.25
    score_mins: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    train_frac_days: float = 0.70

    # IMPORTANT: prevent RAM blowups in XGB step
    xgb_max_days: int = 30
    xgb_usecols: Tuple[str, ...] = ( # read only needed cols when loading df0 grids
        "bid_px_00","ask_px_00","bid_sz_00","ask_sz_00",
        "net_aggr_frac","aggr_frac",
        "sell_at_bid_1s",
    )

    # overwrite controls
    overwrite_norm: bool = False
    overwrite_grid: bool = False