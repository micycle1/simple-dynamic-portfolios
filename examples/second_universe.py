"""A five-asset universe run end to end on prepared pandas objects.

Nothing here is paper-specific: synthetic total-return prices, arbitrary
feature names, a baseline that omits one investable asset, an initial
portfolio, and both built-in and externally prepared model inputs.

Run from the repository root::

    uv run python examples/second_universe.py
"""

import numpy as np
import pandas as pd

import sbg_portfolios as sbg

ASSETS = ["equity_us", "equity_em", "treasury", "credit", "commodity"]
ANNUAL_VOLS = np.array([0.16, 0.22, 0.06, 0.08, 0.20])
CORRELATION = np.array([
    [1.0, 0.7, -0.2, 0.3, 0.2],
    [0.7, 1.0, -0.1, 0.4, 0.3],
    [-0.2, -0.1, 1.0, 0.5, 0.0],
    [0.3, 0.4, 0.5, 1.0, 0.1],
    [0.2, 0.3, 0.0, 0.1, 1.0],
])


def synthetic_inputs(n_days: int = 3000, seed: int = 7) -> tuple[sbg.MarketInputs, pd.DataFrame]:
    """Correlated daily returns with a slowly varying, partly predictable drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days, name="date")
    daily_cov = np.outer(ANNUAL_VOLS, ANNUAL_VOLS) * CORRELATION / 252
    regime = np.cumsum(rng.normal(0.0, 0.02, n_days))  # observable macro state
    drift = 0.04 / 252 + 0.02 / 252 * np.tanh(regime)[:, None] * np.array([1, 1.5, -1, 0, 0.5])
    returns = drift + rng.multivariate_normal(np.zeros(len(ASSETS)), daily_cov, size=n_days)
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=ASSETS)
    # The emerging-market fund launches later; its missing history only affects warm-up.
    prices.iloc[:150, 1] = np.nan
    cash = pd.Series((1 + 0.02) ** (1 / 252) - 1, index=dates)

    features = pd.DataFrame(index=dates)
    features["macro_state"] = regime
    for asset in ASSETS:
        features[f"trend_126d[{asset}]"] = prices[asset].pct_change(126, fill_method=None)
    return sbg.MarketInputs(prices, cash), features


def ewma_covariance(prices: pd.DataFrame, halflife: int = 63) -> sbg.CovarianceHistory:
    """An externally prepared covariance history (here, an EWMA estimate)."""
    covariances = prices.pct_change(fill_method=None).dropna().ewm(halflife=halflife).cov()
    covariances.index.names = ["date", "asset"]
    return sbg.CovarianceHistory(covariances.dropna())


def run_example(n_days: int = 3000) -> dict[str, sbg.StrategyResult]:
    """Forecast, estimate risk, allocate, simulate, and evaluate."""
    market, features = synthetic_inputs(n_days)
    start = market.total_return_prices.index[400]  # after model warm-up
    baseline = sbg.StaticBaseline(pd.Series({
        "equity_us": 0.40, "equity_em": 0.10, "treasury": 0.35, "credit": 0.15,
    }))  # commodity is investable but has a zero baseline weight
    initial = sbg.InitialPortfolio.from_values(
        pd.Series({"equity_us": 60_000.0, "treasury": 30_000.0}), cash_value=10_000.0)
    config = sbg.MarkowitzConfig(annual_vol_target=0.08, relative_l1_radius=0.5,
                                rebalance_frequency="M", objective_horizon_days=21)

    results = {
        "ridge + rolling risk": sbg.run_markowitz_with_models(
            market, baseline,
            alpha_model=sbg.RidgeAlpha(horizon_days=21, max_history=504, min_window=126),
            risk_model=sbg.RollingSampleCovariance(window=63),
            features=features, config=config, initial_portfolio=initial, start=start),
        "ewma + prepared covariance": sbg.run_markowitz(
            market, baseline,
            alpha=sbg.EwmaAlpha(halflife=126).forecast(market.total_return_prices),
            risk=ewma_covariance(market.total_return_prices),
            config=config, initial_portfolio=initial, start=start),
        "baseline": sbg.run_fixed_weight(market, baseline, rebalance_frequency="Q", start=start),
        "baseline vol-controlled": sbg.run_vol_controlled_fixed_weight(
            market, baseline, annual_vol_target=0.08, vol_window=63, start=start),
    }
    return results


def main() -> None:
    """Print the evaluation table and average allocations."""
    results = run_example()
    cash = synthetic_inputs()[0].cash_returns
    table = pd.DataFrame({name: r.evaluate(cash_reference=cash, cash_label="cash")
                          for name, r in results.items()}).T
    print(table.to_string(float_format="{:.4f}".format))
    print("\nAverage allocation (including cash):")
    print(pd.DataFrame({name: r.backtest.composition.mean() for name, r in results.items()})
          .T.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
