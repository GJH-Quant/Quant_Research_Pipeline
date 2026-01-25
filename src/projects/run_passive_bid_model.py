from pathlib import Path
from passive_bid_model.config import Config
from passive_bid_model.pipeline import run_pipeline

cfg = Config(
    raw_trades_dir=Path(r"Dir"),
    raw_mbp1_dir=Path(r"Dir"),
    norm_trades_dir=Path(r"Dir"),
    norm_mbp1_dir=Path(r"Dir"),
    grid_dir=Path(r"Dir"),
    bar_freq="1s",
    overwrite_norm=False,
    overwrite_grid=False,
    run_xgb=True,
    xgb_max_days=200,
)

out = run_pipeline(cfg)
print(out["sweep"].head())
print(out["ev_all"].head())
print(out["xgb_gating"].head())