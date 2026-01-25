# =========================
# FILE: pipeline.py
# =========================
from __future__ import annotations

import pathlib
from typing import Dict, List

import pandas as pd

from .config import Config
from . import data
from .ev import compute_ev_variants
from .modeling import run_xgb_score_gating


def run_pipeline(cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Full end-to-end:
      raw trades+mbp1 -> normalized parquet -> freq-grid per day -> df0 per day -> pipeline summaries

    Memory-safe design:
      - processes per day (read daily trades/mbp1 only)
      - computes sell@bid from trades parquet in chunks (pyarrow)
      - writes df0 grids to disk (optional)
      - XGB loads ONLY last N days + ONLY required columns
    """
    # 1) raw -> normalized (respect overwrite flag)
    trades_parqs, mbp1_parqs = data.normalize_raw_dirs(
        raw_trades_dir=cfg.raw_trades_dir,
        raw_mbp1_dir=cfg.raw_mbp1_dir,
        norm_trades_dir=cfg.norm_trades_dir,
        norm_mbp1_dir=cfg.norm_mbp1_dir,
        tz=cfg.tz,
        rth_start=cfg.rth_start,
        rth_end=cfg.rth_end,
        overwrite=cfg.overwrite_norm,
    )

    # Map by "day key" extracted from filename stem
    # Assumes stems match between trades and mbp1.
    def key(p: pathlib.Path) -> str:
        return p.name.split(".")[0]

    trades_map = {key(p): p for p in trades_parqs}
    mbp1_map = {key(p): p for p in mbp1_parqs}
    common_days = sorted(set(trades_map) & set(mbp1_map))
    if not common_days:
        raise RuntimeError("No overlapping normalized trades/mbp1 parquet days found.")

    sweep_rows: List[dict] = []
    ev_rows: List[pd.DataFrame] = []

    # if you later want these, you can return them; keeping them off by default avoids memory growth
    diag_spread_tables: Dict[str, pd.DataFrame] = {}
    diag_raw_tables: List[pd.DataFrame] = []  # FIX: correct type

    # 2) per day
    for dkey in common_days:
        trades_p = trades_map[dkey]
        mbp1_p = mbp1_map[dkey]

        # day-by-day loads (reasonable)
        trades = pd.read_parquet(trades_p).sort_index()
        mbp1 = pd.read_parquet(mbp1_p).sort_index()
        if trades.empty or mbp1.empty:
            continue

        day = trades.index[0].normalize()

        # 2a) grids
        bbo_grid = data.mbp1_to_grid(
            mbp1,
            day=day,
            rth_start=cfg.rth_start,
            rth_end=cfg.rth_end,
            freq=cfg.bar_freq,
        )
        bars = data.trades_to_tradebars_grid(
            trades,
            day=day,
            rth_start=cfg.rth_start,
            rth_end=cfg.rth_end,
            freq=cfg.bar_freq,
        )
        if bars.empty or bbo_grid.empty:
            continue

        df0 = data.build_df0_grid(bars, bbo_grid)

        # 2b) sell@bid (chunked from daily trades parquet)
        sell_at_bid = data.sell_at_bid_from_trades_parquet_chunked(
            trades_parquet=trades_p,
            bbo_grid=bbo_grid,
            tick=cfg.tick,
            freq=cfg.bar_freq,
            batch_rows=2_000_000,
        )
        df0 = data.attach_sell_at_bid(df0, sell_at_bid)

        # 2c) write df0 grid to disk (optional)
        if cfg.grid_dir is not None:
            data.ensure_dir(cfg.grid_dir)
            out_df0 = cfg.grid_dir / f"{dkey}.df0_{cfg.bar_freq}.parquet"
            if (not out_df0.exists()) or cfg.overwrite_grid:
                df0.to_parquet(out_df0)

        # 3) sweep + EV (computed per day; only summaries kept)
        for jf in cfg.join_fracs:
            df = data.label_passive_bid_fill(df0, H=cfg.H, my_size=cfg.my_size, join_frac=jf, rth_end=cfg.rth_end)

            sweep_rows.append({
                "day": dkey,
                "join_frac": float(jf),
                "n_rows": int(len(df)),
                "can_quote_rate": float(df["can_quote"].mean()),
                "fill_rate": float(df["fill_bid_H"].mean()),
            })

            ev = compute_ev_variants(df, H=cfg.H, cost_bps=0.0)
            ev.insert(0, "day", dkey)
            ev.insert(1, "join_frac", float(jf))
            ev_rows.append(ev)

        # 4) diagnostics (optional; can get big—keeping but not returned unless you want)
        # If you want to truly “stream”, you can comment this whole block out.
        df_diag = data.label_passive_bid_fill(
            df0, H=cfg.H, my_size=cfg.my_size, join_frac=cfg.diag_join_frac, rth_end=cfg.rth_end
        )
        df_diag = data.add_mid_and_forward_moves(df_diag, horizons=cfg.diag_horizons, rth_end=cfg.rth_end)
        for k in cfg.diag_horizons:
            diag_spread_tables[f"{dkey}_k{k}"] = data.summarize_mid_move_conditional(
                df_diag, k=k, use_spread_units=True
            )
            diag_raw_tables.append(data.summarize_mid_move_conditional(df_diag, k=k, use_spread_units=False))

    sweep = pd.DataFrame(sweep_rows).sort_values(["day", "join_frac"]).reset_index(drop=True)
    ev_all = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame()

    # 5) XGB once at end (STRICT cap + usecols)
    if cfg.run_xgb:
        if cfg.grid_dir is None:
            raise RuntimeError("cfg.run_xgb=True requires cfg.grid_dir (df0 grids must be written).")

        df0_files = sorted(cfg.grid_dir.glob(f"*.df0_{cfg.bar_freq}.parquet"))
        if not df0_files:
            xgb_out = pd.DataFrame()
        else:
            df0_files = df0_files[-int(cfg.xgb_max_days):]  # IMPORTANT: cap days

            usecols = list(cfg.xgb_usecols)
            frames = []
            for p in df0_files:
                frames.append(pd.read_parquet(p, columns=usecols).sort_index())

            df0_all = pd.concat(frames, axis=0).sort_index()

            xgb_out = run_xgb_score_gating(
                df0=df0_all,
                H=cfg.H,
                my_size=cfg.my_size,
                join_frac=cfg.xgb_join_frac,
                toxic_spread_mult=cfg.toxic_spread_mult,
                score_mins=cfg.score_mins,
                train_frac_days=cfg.train_frac_days,
                rth_end=cfg.rth_end,
                label_passive_bid_fill=data.label_passive_bid_fill,
            )
    else:
        xgb_out = pd.DataFrame()

    return {
        "sweep": sweep,
        "ev_all": ev_all,
        "xgb_gating": xgb_out,
    }
