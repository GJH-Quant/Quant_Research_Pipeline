# Volatility State Detection (HMM) — Outputs

This document summarizes the outputs of the **volatility state detection** module, which fits **session-aware Hidden Markov Models (HMMs)** to smooth intraday volatility proxies.

The goal of this module is **state discovery**, not prediction:
- Identify persistent intraday volatility regimes
- Produce stable regime labels suitable for conditioning downstream alpha models
- Avoid short-horizon noise and event-driven spikes (handled separately in burst detection)

---

## Dataset & Bars

- **Instrument:** AAPL
- **Bars:** RTH-only time bars
- **Bar Interval:** 60 seconds
- **Sessions:** 243 trading days
- **Bars / day:** 390 (full RTH grid)

### Bar Sanity Checks

- No bars outside RTH
- Inactive bar rate: **0.12%**
- Fraction of bars with `high == low`: **0.23%**
- All OHLCV fields fully populated after fill
- Returns (`ret`, `log_ret`) contain expected NaNs at session boundaries only

---

## Volatility Proxies Modeled

Two smooth, complementary volatility measures were used:

### 1. EWMA Realized Volatility (`rv_ewma`)
- Computed within-session
- Definition: rv_ewma = sqrt(EWMA(log_ret²))

- Parameters:
- EWMA span = 5
- Minimum periods = 5
- Captures **persistent realized volatility level**

### 2. Log High–Low Range (`log_hl`)
- Definition: log_hl = log(high / low)

- Range-based volatility proxy
- Sensitive to intrabar price expansion
- Independent of close-to-close returns

---

## HMM Configuration

- **Model:** Gaussian HMM
- **Covariance:** Diagonal
- **Number of states:** 3
- **Training:** First 70% of sessions
- **Prediction:** Train + test sessions (out-of-sample applied to full dataset)
- **Session-aware lengths:** Each trading day treated as an independent sequence

---

## HMM Results — EWMA Realized Vol (`rv_ewma`)

### State Frequencies

| State | Bars | Interpretation |
|------:|-----:|----------------|
| 0 | 9,996 | High volatility |
| 1 | 35,132 | Medium volatility |
| 2 | 48,427 | Low volatility |

---

### State Characteristics

#### Inactive Rate
| State | Inactive Rate |
|------:|--------------:|
| 0 | 0.04% |
| 1 | 0.01% |
| 2 | 0.22% |

#### Average Volume
| State | Avg Volume |
|------:|-----------:|
| 0 | 58,812 |
| 1 | 26,930 |
| 2 | 14,324 |

#### Return Volatility (`std(log_ret)`)
| State | Std |
|------:|----:|
| 0 | 0.001847 |
| 1 | 0.000691 |
| 2 | 0.000328 |

#### Average Run Length
| State | Avg Bars |
|------:|---------:|
| 0 | 14.0 |
| 1 | 15.5 |
| 2 | 28.7 |

**Interpretation**
- State 0 = short-lived, high-volatility bursts
- State 2 = long, persistent low-volatility regimes
- Clean monotonic ordering across all diagnostics

---

## HMM Results — Log High/Low (`log_hl`)

### State Frequencies

| State | Bars | Interpretation |
|------:|-----:|----------------|
| 0 | 53,007 | Low range |
| 1 | 34,499 | Medium range |
| 2 | 7,264 | High range |

---

### State Characteristics

#### Inactive Rate
| State | Inactive Rate |
|------:|--------------:|
| 0 | 0.22% |
| 1 | 0.003% |
| 2 | 0.00% |

#### Average Volume
| State | Avg Volume |
|------:|-----------:|
| 0 | 13,729 |
| 1 | 30,978 |
| 2 | 105,820 |

#### Return Volatility (`std(log_ret)`)
| State | Std |
|------:|----:|
| 0 | 0.000364 |
| 1 | 0.000755 |
| 2 | 0.002186 |

#### Average Run Length
| State | Avg Bars |
|------:|---------:|
| 0 | 50.4 |
| 1 | 21.8 |
| 2 | 11.5 |

**Interpretation**
- State 2 corresponds to **large range expansion + heavy volume**
- State 0 dominates quiet, range-compressed periods
- Much stronger separation by volume than `rv_ewma`

---

## Comparison: RV vs Log(H/L)

| Aspect | rv_ewma | log_hl |
|------|--------|--------|
| Focus | Persistent volatility | Intrabar expansion |
| Sensitivity | Smooth, slow | Sharp, range-based |
| Run length | Longer in low-vol | Longer in low-range |
| Volume separation | Moderate | Very strong |

These regimes are **not redundant** and should be treated as **separate conditioning signals**.

---

## Intended Use

This module is designed to generate **state labels**, not trading signals.

Recommended usage:
- Feed `states_rv` and `states_hl` as **categorical features** into:
- Alpha classifiers
- Execution models
- Regime-conditioned strategy logic
- Combine with:
- Burst detectors (volume, aggression, urgency)
- Time-of-day normalized features
- Microstructure signals

**Do NOT**:
- Trade directly on state transitions
- Treat states as predictive without conditioning on direction or execution

---

## Notes on Train/Test Handling

- HMM is trained on first 70% of sessions
- States are inferred on the full dataset using the trained model
- This is **appropriate** for regime labeling used as an input feature
- Any downstream alpha model should still perform its own train/test split

---

## Next Extensions (Optional)

- Joint HMM on `[rv_ewma, log_hl]`
- State-conditioned burst intensity thresholds
- Cross-state transition asymmetry analysis
- Regime-aware execution cost modeling

---

**Status:** ✅ Stable  
**Leakage:** None (session-safe, past-only features)  
**Role in Pipeline:** Structural volatility regime labeling