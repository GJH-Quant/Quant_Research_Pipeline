# Volatility Regime Detection — Outputs

This document summarizes the outputs of the **intraday volatility regime detection** pipeline using Hidden Markov Models (HMMs) on 1-minute RTH bars for **AAPL**.

Two independent volatility proxies are modeled:

1. **EWMA Realized Volatility (`rv_ewma`)**
2. **Range-based Volatility (`log_hl = log(high / low)`)**

Each proxy is fit with a **3-state Gaussian HMM**, using **session-aware sequence lengths** to prevent overnight leakage.

---

## Data Sanity Checks

All diagnostics passed prior to model fitting:

- **Bars per day (RTH):** 390 (exact, no drift)
- **Bars outside RTH:** 0
- **Inactive bars:** 0.12%
- **Flat bars (high == low):** 0.23%
- **No missing OHLCV fields**

This confirms a clean, session-aligned intraday grid suitable for regime modeling.

---

## HMM Regimes — EWMA Realized Volatility (`rv_ewma`)

### State Counts

| State | Observations |
|------:|-------------:|
| 0 | 9,996 |
| 1 | 35,132 |
| 2 | 48,427 |

---

### Summary Statistics by State

| State | Inactive Rate | Avg Volume | Std(log returns) | Avg Run Length (bars) |
|------:|---------------|------------|------------------|-----------------------|
| 0 | 0.040% | 58,812 | 0.001847 | 14.0 |
| 1 | 0.014% | 26,930 | 0.000691 | 15.5 |
| 2 | 0.219% | 14,324 | 0.000328 | 28.7 |

---

### Interpretation

- **State 2**  
  - Lowest realized volatility  
  - Lowest volume  
  - Longest persistence  
  - Interpretable as a *quiet / low-energy* regime

- **State 1**  
  - Medium volatility and volume  
  - Transitional regime

- **State 0**  
  - Highest realized volatility  
  - Shortest run length  
  - Likely *active / event-driven* periods

---

## HMM Regimes — Range-based Volatility (`log_hl`)

### State Counts

| State | Observations |
|------:|-------------:|
| 0 | 53,007 |
| 1 | 34,499 |
| 2 | 7,264 |

---

### Summary Statistics by State

| State | Inactive Rate | Avg Volume | Std(log returns) | Avg Run Length (bars) |
|------:|---------------|------------|------------------|-----------------------|
| 0 | 0.215% | 13,729 | 0.000364 | 50.4 |
| 1 | 0.003% | 30,978 | 0.000755 | 21.8 |
| 2 | 0.000% | 105,820 | 0.002186 | 11.5 |

---

### Interpretation

- **State 0**  
  - Tight ranges  
  - Low volume  
  - Extremely persistent  
  - Captures *structural compression / liquidity absorption*

- **State 2**  
  - Wide ranges  
  - Very high volume  
  - Short-lived  
  - Captures *explosive volatility / impulse moves*

- **State 1**  
  - Intermediate regime

---

## Key Takeaways

- The two volatility proxies capture **different dimensions of intraday regime structure**:
  - `rv_ewma` emphasizes **return dispersion**
  - `log_hl` emphasizes **price range expansion**

- Regime assignments are **non-monotonic by design** (HMM state labels are arbitrary).

- Run-length statistics confirm:
  - Low-vol regimes are **persistent**
  - High-vol regimes are **short and bursty**

- These regimes are **ideal categorical features** for downstream models (e.g. XGBoost):
  - Can be **one-hot encoded**
  - Can interact with time-of-day features
  - No assumptions of linearity required

---

## Intended Usage

This module is designed as a **feature-generation layer**, not a trading strategy by itself.

Typical downstream usage:
- One-hot encoded regime states
- Interaction with order-flow, volume, or signal-strength features
- Conditioning execution or entry logic on regime

---

## Notes

- All calculations are **RTH-only**
- No time-of-day normalization was applied intentionally
- No forward information leakage (session-aware HMM fitting)

---