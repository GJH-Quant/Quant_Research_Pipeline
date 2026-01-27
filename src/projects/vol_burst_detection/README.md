# Vol Burst Detection — Outputs

This module builds **intraday burst / intensity features** from RTH time bars. These features are **conditioning signals**, not standalone alphas. They are designed to capture *relative abnormal activity* after time‑of‑day normalization.

---

## Data & Bars

* **Universe**: Equity trades (example run: AAPL)
* **Session**: RTH only (09:30–16:00 NY)
* **Bar type**: Fixed time bars (default: 60s)
* **Grid**: Session‑anchored, fully filled

Sanity checks ensure:

* No bars outside RTH
* Constant bars/day (390 for 60s)
* Explicit inactive bars (zero trades)

Example diagnostics:

```
bars/day summary (filled):
count    243
mean     390
inactive rate: 0.12%
fraction(high==low): 0.23%
```

---

## Burst Features

All burst features are **time‑of‑day normalized** using a `tod_bin` aligned to the bar interval. Normalization is done *within each TOD bin* using only **past data** (shifted by one bar).

### 1. Rolling Percentile Burst Intensity

**Column:** `burst_pct_score`

**Definition:**

* For each TOD bin:

  * Compute a rolling historical quantile of the feature (default: 99.5%)
  * Compare current value to this threshold
* Output is a *smooth intensity*, not a binary event

Mathematically:

```
thr_t = rolling_quantile(x_{t-1}, q)
burst_pct_score = log1p(x_t / thr_t)
```

**Properties:**

* Scale‑free
* Stable across regimes
* Interpretable as "how extreme vs recent history"

**Observed distribution (example):**

```
non-null rate: 91.77%
median: 0.18
99th pct: 0.86
99.5th pct: 1.01
approx event rate (score > 1.0): ~2.2%
```

This aligns with design intent: *rare but meaningful bursts*.

---

### 2. Median / MAD Robust Z‑Score

**Column:** `burst_mad_z`

**Definition:**

* For each TOD bin:

  * Compute rolling median
  * Compute rolling MAD (median absolute deviation)
  * Scale MAD by 1.4826

```
burst_mad_z = (x_t − median) / (MAD * 1.4826)
```

**Properties:**

* Robust to fat tails
* Interpretable as a statistical surprise
* More sensitive than percentile method

**Observed distribution (example):**

```
non-null rate: 83.95%
median: ~0.02
95th pct: ~4.6
99th pct: ~10.9
approx event rate (z > 3): ~9.1%
```

This feature is intentionally **higher‑frequency** than the percentile burst.

---

## Why Two Burst Signals?

They serve **different roles**:

| Feature           | Best Use                                  |
| ----------------- | ----------------------------------------- |
| `burst_pct_score` | Regime‑stable conditioning, gating trades |
| `burst_mad_z`     | Short‑horizon aggressiveness / urgency    |

They are **not meant to agree**.

---

## Usage in Alpha Models

Recommended usage patterns:

* Conditioning filters (e.g. *only trade if burst_pct_score > X*)
* Feature inputs to classifiers (tree / linear)
* Joint conditioning with:

  * volatility regimes
  * order‑flow imbalance
  * liquidity state

**Not recommended:**

* Using burst signals alone as direction
* Threshold‑only strategies without downstream logic

---

## Leakage & Safety

* All statistics are computed using **strictly lagged data**
* TOD normalization prevents intraday seasonality leakage
* Safe to compute on full dataset **after** model fitting

This module is designed to be used upstream of:

* directional classifiers
* execution / fill models
* regime‑aware alpha stacks

---

## Output Columns

Final bars include:

* `burst_pct_score`
* `burst_mad_z`
* (optional) `tod_bin`

No intermediate thresholds or raw stats are retained by design.

---

## Status

✔ Production‑ready feature block
✔ Safe for research + live conditioning
✔ Stable under regime shifts

Next logical extensions:

* Winsorized MAD z
* Multi‑scale burst (30s / 60s / 5m)
* Joint burst × volatility state interaction terms
