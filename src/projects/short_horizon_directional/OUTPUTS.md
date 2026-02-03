# Outputs — HL-Gated Tail Event Classifier (AAPL, 2s)

This document summarizes the **TEST-set results** for the AAPL 2-second tail-event classifier, including overall metrics and **PnL proxy diagnostics**, both **ungated** and **HL-regime gated**.

---

## Model Metrics (TEST)

| Metric | Value |
|------|------|
| Train rows | 1,813,890 |
| Test rows | 795,595 |
| Positive rate (train) | 12.07% |
| Positive rate (test) | 8.55% |
| Scale pos weight | 7.29 |
| AUC | **0.6442** |
| Average Precision | **0.1451** |
| Log loss | 0.5906 |
| Brier score | 0.2011 |

---

## PnL Proxy (No Gate)

Top-quantile selection based on model score (`q = 0.80`, i.e. top 20%).

| Metric | Value |
|------|------|
| Rows evaluated | 795,590 |
| Score threshold | 0.5303 |
| Top rows selected | 159,118 |
| Mean fwd Δmid | **0.00139** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 47.89% |

**Interpretation**
- Positive mean forward mid-move despite hit rate < 50%, indicating **positive skew / tail payoffs**.
- This is a **diagnostic proxy** (no costs, no execution).

---

## PnL Proxy (HL-Gated)

Same top-quantile logic applied **within each HL volatility regime**.

### HL State 0 (Low HL)

| Metric | Value |
|------|------|
| Rows | 526,080 |
| Score threshold | 0.5203 |
| Top rows | 105,216 |
| Mean fwd Δmid | **0.00019** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 46.86% |

**Observation**
- Very weak edge; near-zero mean drift.
- Most likely will need to capture spread in order to gain positive EV after costs. 

---

### HL State 1 (Medium HL)

| Metric | Value |
|------|------|
| Rows | 238,740 |
| Score threshold | 0.5452 |
| Top rows | 47,748 |
| Mean fwd Δmid | **0.00332** |
| Median fwd Δmid | 0.00000 |
| Hit rate | 49.01% |

**Observation**
- Clear improvement vs ungated and HL-0.

---

### HL State 2 (High HL)

| Metric | Value |
|------|------|
| Rows | 30,770 |
| Score threshold | 0.5898 |
| Top rows | 6,154 |
| Mean fwd Δmid | **0.00747** |
| Median fwd Δmid | **0.00500** |
| Hit rate | **50.89%** |

**Observation**
- Strongest regime by far:
  - Highest mean and median forward move
  - Hit rate > 50%
- Confirms the signal is **high-volatility / impulse-driven** and **state-dependent**.

---

## Important Caveats

- PnL proxy uses **raw mid-price drift only**:
  - No spread, slippage, or fill assumptions.
- Statistics are **diagnostic**, not deployable P&L.
- Next step would be to model fills to see if fills can happen with minimal adverse selection.