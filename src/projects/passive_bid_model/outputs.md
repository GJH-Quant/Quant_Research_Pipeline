# Example Outputs — Passive Bid Model (AAPL)

This file shows representative outputs from the **AAPL passive bid execution model** using:
- Trade prints
- MBP-1 (top-of-book)
- 1-second RTH grids

Results are shown for **one liquid equity session (2025-01-02)** to validate modeling logic.
They are **not claims of profitability**.

---

## 1. Fill Rate vs Queue Position

Queue position is approximated as a fraction of displayed bid size:

queue_ahead = join_frac × bid_size

| Join Fraction | Fill Rate |
|--------------|-----------|
| 0.50 | 23.6% |
| 0.75 | 20.8% |
| 1.00 | 18.3% |
| 1.25 | 16.2% |
| 1.50 | 14.8% |

**Observation**
- Fill probability decreases monotonically with queue depth
- Confirms expected limit order queue dynamics

---

## 2. Execution EV (Conditional on Fill)

EV is reported **per quote opportunity**, conditional on fills.

| Join Fraction | Exit Type | EV / Quote ($) |
|--------------|----------|----------------|
| 0.50 | Exit @ Bid | -0.0021 |
| 0.50 | Exit @ Mid | +0.0011 |
| 1.00 | Exit @ Bid | -0.0019 |
| 1.00 | Exit @ Mid | +0.0012 |

**Observation**
- Exit-at-bid EV is negative due to adverse selection
- Exit-at-mid provides an optimistic upper bound on markout

---

## 3. Model-Based Quote Gating (XGBoost)

Two models estimate:
- Fill probability
- Toxicity (adverse selection risk)

Quotes are gated using:

score = P(fill) × (1 − P(toxic))

| Score Threshold | Quote Rate | Fill Rate | EV / Quote (Mid) |
|----------------|-----------|----------|------------------|
| 0.10 | 99.5% | 13.9% | +0.00058 |
| 0.20 | 65.1% | 17.4% | +0.00076 |
| 0.25 | 32.6% | 22.8% | +0.00124 |
| 0.30 | 12.3% | 30.8% | +0.00259 |

**Observation**
- Gating reduces participation but improves execution quality
- EV per quote increases monotonically with stricter thresholds

---

## Assumptions

This model makes the following explicit assumptions:

### Market Data
- Uses MBP-1 (top-of-book only) quotes; full order-book depth is not modeled
- Trade aggressor side is inferred from exchange prints
- Quotes are assumed stable within each 1-second bin

### Queue Position
- Queue position is approximated as a fraction of displayed bid size  
- All displayed size ahead is assumed to have priority
- Hidden liquidity and midpoint pegs are not modeled

### Fill Logic
- A bid is considered filled if cumulative sell-at-bid volume exceeds (queue_ahead + own_order_size)
- Partial fills are treated as full fills
- Order cancellations, re-queuing, and latency effects are ignored

### Execution & EV
- **Exit @ Mid** represents an optimistic upper bound on achievable markout
- **Exit @ Bid** ignores exchange fees, rebates, and transaction costs
- No explicit inventory or risk limits are modeled

### Modeling
- Fill and toxicity are modeled independently using XGBoost classifiers
- Models are trained and evaluated using day-based splits
- No hyperparameter optimization beyond reasonable defaults

---

## Limitations

- Results shown are from a **single highly liquid equity session (AAPL)**
- No multi-day PnL or distributional robustness analysis shown here
- No transaction costs, fees, or rebates applied
- No depth-aware queue modeling (MBP-10)
- Exit logic is simplified and does not reflect real-time order management

---

## Intended Use

This framework is designed for:
- Studying **passive execution quality**
- Understanding queue position vs adverse selection trade-offs
- Demonstrating execution-aware modeling techniques

It is **not** intended to represent a production-ready or live-trading strategy