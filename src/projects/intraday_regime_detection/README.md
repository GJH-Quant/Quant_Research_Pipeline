# Intraday Regime Detection with Hidden Markov Models

## Objective
Identify and validate **intraday volatility and liquidity regimes** using high-frequency equity data, with an emphasis on **data hygiene, regime persistence, and train/test stability** rather than direct PnL optimization.

This project demonstrates an end-to-end quantitative research workflow focusing on robustness and interpretability instead of backtest-driven overfitting.

## Overview

This study applies **Hidden Markov Models (HMMs)** to 1-minute intraday bars to classify market regimes using two complementary features:

1. **Log High/Low Volatility (`log(H/L)`)**
2. **Time-of-Day (TOD) Normalized Volume** using robust median/MAD normalization

Each regime model is evaluated on:
- Interpretability of states  
- Regime persistence and transition dynamics  
- Out-of-sample (train/test) stability  
- Absence of lookahead bias  

The goal is to learn **structural intraday regimes**, not to optimize a trading strategy.

## Data

- **Source:** NASDAQ TotalView / ITCH trades (via Databento)
- **Instrument:** ABNB (example equity)
- **Granularity:** Raw trades → 1-minute bars
- **Session:** Regular Trading Hours (09:30–16:00 ET)
- **Inactive bars:** Forward-filled price, zero volume, flagged explicitly

Raw data files and parquet outputs are intentionally excluded from the repository.