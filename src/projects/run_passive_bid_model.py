from pathlib import Path
from passive_bid_model.config import Config
from passive_bid_model.pipeline import run_pipeline

cfg = Config(
    raw_trades_dir=Path(r"D:\Quant_Research_Pipeline\data\00_raw\databento\equities\aapl\trades"),
    raw_mbp1_dir=Path(r"D:\Quant_Research_Pipeline\data\00_raw\databento\equities\aapl\mbp1"),
    norm_trades_dir=Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\trades"),
    norm_mbp1_dir=Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\mbp1"),
    grid_dir=Path(r"D:\Quant_Research_Pipeline\data\01_normalized\equities\aapl\1s_grid"),
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