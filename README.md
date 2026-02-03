# Quant Research Pipeline

A modular quantitative research framework focused on intraday market microstructure,
order-book dynamics, and regime detection using high-resolution Databento data
(Trades, MBP-1, MBP-10).

The goal of this repository is to support systematic research, feature engineering,
and strategy development for intraday trading and execution-alpha style strategies,
with an emphasis on robustness, reproducibility, and clean data pipelines.

---

## Core Focus Areas
- Intraday market microstructure analysis
- Order-book–based features (spread, depth, imbalance, OFI)
- Volatility and regime detection
- Non-directional and conditional-distribution strategies

---

## Repository Structure

### Data & Normalization

- `src/loaders/`  
  Utilities for loading, cleaning, validating, and converting raw market data  
  (trades, MBP-10) into analysis-ready Parquet files and DataFrames

---

### Feature Engineering
- `src/bars/`  
  Bar construction modules:
  - time bars
  - volume bars
  - dollar bars
  - signed-imbalance bars

- `src/features/`  
  Feature generation from bars and order-book snapshots

- `src/labels/`  
  Forward return, volatility, and distribution-shape labels

---

### Projects
- `src/projects/`  
  Self-contained research projects and experiments  
  Each project typically includes:
  - its own script(s)
  - project-specific README
  - outputs and results

---

## Design Principles
- Strict timestamp normalization (NYSE RTH, timezone-aware)
- Deterministic ordering (event time + sequence)
- Explicit data validation and sanity checks
- Separation of data, features, labels, and models
- Research-first, production-aware architecture

---

## Dependencies
- Python 3.10+
- pandas
- numpy
- databento
- hmmlearn
- scikit-learn

---

## Notes
This repository is research-oriented and intentionally avoids over-optimization.
Results are evaluated across multiple regimes, time-of-day segments, and volatility
states to minimize false discovery and overfitting.