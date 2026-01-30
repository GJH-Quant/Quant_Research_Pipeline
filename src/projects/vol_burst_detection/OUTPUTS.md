# Output Summary – Volume Burst Features

This document summarizes the outputs produced by the volume burst detection pipeline.

---

## Dataset Overview

- Bars: ~2.84M (2s RTH bars)
- Sessions: ~243 trading days
- Inactive bars: ~15.6%
- Bars/day: 11,700 (expected for 2s RTH)

---

## MED / MAD Normalization (`log_volume_mad_z`)

**Purpose:**  
Measures how unusual current volume is *relative to its normal behavior at the same time of day*.

**Key Stats (10-day lookback):**
- NaNs: ~8.2% (expected warmup)
- Median: ~0
- 99%: ~1.7
- 99.9%: ~2.6
- Max: ~8.3

**Interpretation:**
- Values near 0 → normal activity
- >2 → unusually high volume
- Negative values → suppressed liquidity

---

## TOD Burst Score (`log_volume_pct_score`)

**Purpose:**  
Continuous intensity measure of volume vs rolling TOD baseline.

**Key Stats:**
- NaNs: ~4.1%
- Median: ~0.69 (≈ log(2))
- 99%: ~0.96
- 99.9%: ~1.06
- Max: ~1.79

**Interpretation:**
- ~0.69 → ~2× typical volume
- ~1.1 → ~3× typical volume
- High values correspond to genuine burst activity

---

## Notes

- No quantile thresholds are used
- No binary events are created
- All features are continuous and ML-ready
- NaNs occur only during warmup or missing baseline windows