# Intraday Volume Burst Detection (TOD-Normalized)

This project builds **intraday volume burst features** using **time-of-day (TOD)–conditioned rolling statistics** on regularized time bars.  
The goal is to normalize volume behavior **relative to its typical behavior at the same time of day**, enabling regime-aware and execution-relevant signals.

---

## What This Does

1. **Builds RTH time bars** (e.g. 2s) from trades
2. **Computes log-volume**
3. **Applies TOD-based MED/MAD normalization**
   - Robust z-score per TOD bucket
   - Rolling lookback measured in *days*
4. **Computes a TOD baseline burst score**
   - Ratio of current volume vs rolling TOD baseline
   - Log-scaled for stability

No labels or events are created — all outputs are **continuous features** suitable for ML or conditional analysis.

---

## Key Features

- `log_volume`
- `log_volume_mad_z`  
  Robust deviation from typical TOD behavior
- `log_volume_pct_score`  
  Continuous burst intensity vs TOD baseline

---

## Design Principles

- **No lookahead** (all rolling stats are shifted)
- **Session-aware** (RTH only, per-day resets)
- **Robust to outliers** (median / MAD)
- **Intraday-specific** (TOD buckets, not global stats)