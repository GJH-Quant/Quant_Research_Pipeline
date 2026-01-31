# README.md

## Volatility State Detection (HMM)

Builds 60s RTH bars from trade prints, computes two intraday volatility features, and fits **1D Gaussian HMM** regimes for each feature with a clean **session-based train/test split** (no lookahead across days).

### What it does
1. Load normalized trades (parquets)
2. Create **60s RTH bars** with basic aggressive-flow fields (buy/sell/neutral)
3. Compute:
   - `rv_ewma`: EWMA realized vol of `log_ret`
   - `log_hl`: log(high/low) range
4. Split by `session_date` (`TRAIN_FRAC` train, remainder test)
5. Fit HMM regimes separately on:
   - `rv_ewma` → `states_rv`
   - `log_hl`  → `states_hl`
6. Print summary diagnostics (counts, inactive rate, volume, std(log_ret), mean feature, run length)