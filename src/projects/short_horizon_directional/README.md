# HL-Gated Tail Event Classifier (AAPL, 2s Bars)

This script builds and evaluates a short-horizon **mid-price tail event classifier** for AAPL using **2-second time bars with MBP-1 (BBO)** data. It reports standard **test-set classification metrics** and a simple **PnL proxy**, both **ungated** and **gated by HL volatility regime**.

---

## What the script does

1. **Load data**
   - Trades and MBP-1 (BBO) parquet data for AAPL from 2025-01-01 --> 2025-12-20.

2. **Build bars**
   - Construct 2-second bars with BBO fields using `create_time_bars_w_bbo`.

3. **Volatility regimes**
   - Run `vol_state_detection` and merge results.
   - Create one-hot HL regime indicators:
     - `states_hl_0`, `states_hl_1`, `states_hl_2`

4. **Feature engineering**
   - Volume features: `log_volume_mad_z`
   - Microstructure / flow:
     - `imbalance`
     - `spread`
     - `bbo_updates`
     - `mp_over_spread = microprice_diff / spread`
     - `net_aggr = aggr_buy_vol - aggr_sell_vol`
    (these were selected based on which combinations had the highest AP)

5. **Labeling**
   - Build tail labels with `build_tail_labels`:
     - `y_up`, `y_down`
     - Horizon = `HORIZON`
     - Threshold = `SPREAD_MULT × spread`

6. **Model training**
   - Train an **XGBoost classifier** with a session-based train/test split.
   - Handle class imbalance via `scale_pos_weight`.

7. **Evaluation**
   - Report TEST metrics: AUC, AP, logloss, Brier.
   - Compute a **PnL proxy** on TEST timestamps:
     - `fwd_dmid = mid[t + H] - mid[t]`
     - Select top tail predictions by probability quantile (`Q_PROBA_THRESH`).
   - Report PnL proxy:
     - Ungated (all TEST data)
     - Gated separately for each HL state (`0 / 1 / 2`)