# outputs.md

## Run Summary (AAPL, 60s RTH bars)

### Bar construction sanity
- Sessions: `243`
- Bars/day: `390` (RTH 09:30–16:00, 60s)
- Bars outside RTH: `0`
- Inactive rate: `0.12%`
- Fraction(high == low): `0.23%`

---

## HMM regimes on `rv_ewma` → `states_rv`
### State counts
- 0: `9,996` (~10%)
- 1: `35,132` (~35%)
- 2: `48,427` (~55%)

### Per-state diagnostics
- Inactive rate: 0=`0.040%`, 1=`0.014%`, 2=`0.219%`
- Avg volume: 0=`58.8k`, 1=`26.9k`, 2=`14.3k`
- Std(log_ret): 0=`0.001847`, 1=`0.000691`, 2=`0.000328`
- Mean `rv_ewma`: 0=`0.001611`, 1=`0.000679`, 2=`0.000329`
- Avg run length (bars): 0=`14.0`, 1=`15.5`, 2=`28.7`

Interpretation: state 0=high vol / high activity, state 2=low vol / sticky.

---

## HMM regimes on `log_hl` → `states_hl`
### State counts
- 0: `53,007` (~61%)
- 1: `34,499` (~39%)
- 2: `7,264` (~8%)

### Per-state diagnostics
- Inactive rate: 0=`0.215%`, 1=`0.003%`, 2=`0.000%`
- Avg volume: 0=`13.7k`, 1=`31.0k`, 2=`105.8k`
- Std(log_ret): 0=`0.000364`, 1=`0.000755`, 2=`0.002186`
- Mean `log_hl`: 0=`0.000532`, 1=`0.001136`, 2=`0.003040`
- Avg run length (bars): 0=`50.4`, 1=`21.8`, 2=`11.5`

Interpretation: state 0=compression, state 2=range expansion / event bars.