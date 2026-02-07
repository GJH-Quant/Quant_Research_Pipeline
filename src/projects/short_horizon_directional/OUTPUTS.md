# Outputs — HL-Gated Tail Event Classifier (AAPL, 2s)

This document summarizes the **TEST-set results** for the AAPL 2-second tail-event classifier, including overall metrics, **PnL proxy diagnostics**, and supporting **sanity checks + HMM regime diagnostics** used for HL gating.

---

## Sanity Checks

### 2s Bars (RTH)

- Bars/day (filled): **11,700** (243 sessions; std=0)
- **OK: no bars outside RTH**
- Inactive rate: **15.63%**
- Fraction(high == low): **37.91%**

### HL Regime Bars (post-regime build)

- Bars/day (filled): **390** (243 sessions; std=0)
- **OK: no bars outside RTH**
- Inactive rate: **0.12%**
- Fraction(high == low): **0.23%**

**Interpretation**
- Dense 2s sampling naturally produces a large fraction of “no-range” bars (high == low).
- The HL-regime representation is materially cleaner (lower inactive + high==low), so gating on HL regimes is mechanically reasonable.

---

## HMM Regimes

### HMM Regimes on `rv_ewma`

**State counts**
- State 0: **9,996**
- State 1: **35,132**
- State 2: **48,427**

**Per-state diagnostics**

- Inactive rate:
  - S0: 0.000400
  - S1: 0.000142
  - S2: 0.002189

- Average volume:
  - S0: 58,811.7865
  - S1: 26,929.8485
  - S2: 14,324.4074

- Std(log_ret):
  - S0: 0.001847
  - S1: 0.000691
  - S2: 0.000328

- Average `rv_ewma`:
  - S0: 0.001611
  - S1: 0.000679
  - S2: 0.000329

- Average run length (bars):
  - S0: 14.0000
  - S1: 15.5040
  - S2: 28.7401

---

### HMM Regimes on `log_hl` (HL gating)

**State counts**
- HL State 0: **53,007**
- HL State 1: **34,499**
- HL State 2: **7,264**

**Per-state diagnostics**

- Inactive rate:
  - HL0: 0.002151
  - HL1: 0.000029
  - HL2: 0.000000

- Average volume:
  - HL0: 13,729.3516
  - HL1: 30,977.7260
  - HL2: 105,820.1323

- Std(log_ret):
  - HL0: 0.000364
  - HL1: 0.000755
  - HL2: 0.002186

- Average `log_hl`:
  - HL0: 0.000532
  - HL1: 0.001136
  - HL2: 0.003040

- Average run length (bars):
  - HL0: 50.4348
  - HL1: 21.8486
  - HL2: 11.5485

**Interpretation**
- HL2 is the “impulse” regime: highest volume and return dispersion, but shortest run length.
- HL0 is the quiet regime: lowest vol + lowest volume, longest run length.

---

## Feature QA

### MED/MAD Normalization Check — `log_volume -> log_volume_mad_z`

- Rows: **2,843,100**
- NaN % raw: **0.0**
- NaN % z  : **0.0821673525** (8.2167%)
- Z-score median (should be ~0): **-0.0033849374**
- Z-score 99.9%: **2.5725968087**
- Z-score max  : **8.3265593788**

---

### TOD Burst Check (NO QUANTILE) — `log_volume -> log_volume_pct_score`

- Rows: **2,843,100**
- NaN % raw  : **0.0**
- NaN % score: **0.0411522634** (4.1152%)
- Score median: **0.6931471806**
- Score 99.9%: **1.0659029524**
- Score max  : **1.7931591798**

---

## Model Metrics (TEST)

| Metric | Value |
|------|------|
| Train rows | 1,813,110 |
| Test rows | 795,260 |
| Positive rate (train) | 12.0503% |
| Positive rate (test) | 8.5295% |
| Scale pos weight | 7.298518 |
| AUC | **0.6430** |
| Average Precision | **0.14285** |
| Log loss | 0.591001 |
| Brier score | 0.201330 |

---

## PnL Proxy (No Gate)

Top-quantile selection based on model score (**q = 0.95**, i.e. top 5%).

| Metric | Value |
|------|------|
| Rows evaluated | 795,255 |
| Score threshold | 0.6236296177 |
| Top rows selected | 39,763 |
| Mean fwd Δmid | **0.0059601891** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 48.5326% |

**Interpretation**
- Positive mean forward mid-move despite hit rate < 50% suggests **positive skew / tail contribution**.
- This remains a **diagnostic proxy** (no costs, no execution).

---

## PnL Proxy (HL-Gated)

Same top-quantile logic (**q = 0.95**) applied **within each HL volatility regime**.

### HL State 0 (Low HL)

| Metric | Value |
|------|------|
| Rows | 526,030 |
| Score threshold | 0.6034011304 |
| Top rows | 26,302 |
| Mean fwd Δmid | **0.0010409855** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 47.4070% |

**Observation**
- Weak edge; after costs this likely needs **spread capture / passive entry** to survive.

---

### HL State 1 (Medium HL)

| Metric | Value |
|------|------|
| Rows | 238,460 |
| Score threshold | 0.6572420806 |
| Top rows | 11,923 |
| Mean fwd Δmid | **0.0118917219** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 48.0584% |

**Observation**
- Clear improvement vs HL0 and vs no-gate mean drift, with similar hit rate.

---

### HL State 2 (High HL)

| Metric | Value |
|------|------|
| Rows | 30,765 |
| Score threshold | 0.6700672507 |
| Top rows | 1,539 |
| Mean fwd Δmid | **0.0245938921** |
| Median fwd Δmid | **0.01500** |
| Hit rate | **53.4763%** |

**Observation**
- Strongest regime by far:
  - Highest mean and median forward move
  - Hit rate **> 50%**
- Confirms the signal is **high-volatility / impulse-driven** and **state-dependent**.

---

## Important Caveats

- PnL proxy uses **raw mid-price drift only**:
  - No spread, slippage, or fill assumptions.
- Statistics are **diagnostic**, not deployable P&L.
- Next step is a **fill + cost model** to test whether HL2 survives realistic execution.