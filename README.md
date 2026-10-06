# simple-portfolio

A small, DataFrame-first library for anchored, volatility-limited Markowitz
portfolios, extracted from the reference implementation of
[*Simple Dynamic Stock/Bond/Gold Portfolios*](https://stanford.edu/~boyd/papers/stock_bond_gold_portfolios.html)
([cvxgrp/simple-portfolio-code](https://github.com/cvxgrp/simple-portfolio-code)).

The library accepts prepared pandas inputs and configuration only: no downloads,
file paths, or ticker-specific logic. It contains

- the anchored volatility-limited Markowitz optimizer (CVXPY),
- ridge and EWMA return forecasts,
- a rolling sample covariance estimator, plus adapters for factor or full
  covariance histories,
- an untaxed daily simulator with explicit cash, cost, and timing conventions,
- performance metrics.

## The optimization problem

At each rebalance the strategy solves

    maximize    alpha^T w + r_cash (1 - g) - (s/2) ||w - w_prev||_1
    subject to  w >= 0,  g = 1^T w <= L,
                ||[F^T; diag(d)] w||_2 <= sigma,
                ||w - g a||_1 <= rho g

where `a` is the static baseline, `sigma` the daily volatility limit, `L` the
maximum risky exposure, `s` the bid-ask spread, and `rho` the anchor radius
(`rho = 1` is the paper's constraint and the default).

## Usage

```python
import pandas as pd
import simple_portfolio as sp

result = sp.run_markowitz(
    market=sp.MarketInputs(total_return_prices=prices, cash_returns=daily_cash_returns),
    baseline=sp.StaticBaseline(pd.Series({"stocks": 0.5, "bonds": 0.3, "gold": 0.2})),
    alpha=sp.AlphaForecasts(expected_21d_returns, horizon_days=21),
    risk=sp.CovarianceHistory(daily_covariances),  # or a FactorRiskHistory
    config=sp.MarkowitzConfig(annual_vol_target=0.07, relative_l1_radius=1.0),
    initial_portfolio=None,  # all cash; or sp.InitialPortfolio.from_values(...)
)
result.evaluate(cash_reference=daily_cash_returns)
```

Alpha and risk inputs can instead come from the built-in models, e.g.
`sp.RidgeAlpha(...).forecast(prices, features)` and
`sp.RollingSampleCovariance(window=...).estimate(prices)`, or
`sp.run_markowitz_with_models(...)`. Fixed-weight and volatility-controlled
fixed-weight benchmarks are available as `sp.run_fixed_weight` and
`sp.run_vol_controlled_fixed_weight`.

Conventions:

- An observation labelled with date `t` is used at the rebalance on `t`; the
  library does not lag supplied data.
- Inputs are validated and aligned by label once, to the market's column order.
  Missing forecasts, risk, or cash on a decision date are errors; nothing is
  silently filled.
- A solver status other than `optimal` raises `SolverError` (optionally
  `optimal_inaccurate` can be accepted); there is no fallback portfolio.
- A rebalance day's NAV is recorded before that rebalance's costs, which reach
  the next NAV; the first observation has zero asset and cash return. These
  match the paper's reference results.

[`examples/second_universe.py`](examples/second_universe.py) runs the library
end to end on a synthetic five-asset universe.

## The paper as an example

[`notebooks/paper_example.ipynb`](notebooks/paper_example.ipynb) rebuilds the
paper's SPY/AGG/GLD results step by step with this API: forecasts, risk,
Markowitz and benchmark backtests, the evaluation table, forecast accuracy (the
paper's Figure 7), the constraints at each decision, and variations of anchor
radius, initial portfolio, risk model, and volatility limit. It is saved with
its outputs (interactive plotly figures), so it can be read without running it.

Its input data is not distributed here. Running it needs the `data/` folder
written by `scripts/1_download_data.py` and
`scripts/2_download_evaluation_data.py` in
[cvxgrp/simple-portfolio-code](https://github.com/cvxgrp/simple-portfolio-code),
copied into this repository or pointed to by the `SBG_DATA` environment
variable. `uv sync` installs the notebook's dependencies (ipykernel, plotly,
pyarrow) in the development group.

## Install and test

```bash
uv sync
uv run python -m unittest discover -s tests
uv run python examples/second_universe.py
```
