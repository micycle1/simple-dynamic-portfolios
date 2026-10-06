"""DataFrame-first anchored Markowitz portfolios: forecasts, risk, allocation, simulation.

The package accepts prepared pandas inputs and configuration only: it never
downloads data, reads paper artifacts, or special-cases tickers.
"""

from .alpha import AlphaForecasts, EwmaAlpha, RidgeAlpha
from .baseline import InitialPortfolio, StaticBaseline
from .config import MarkowitzConfig
from .data import MarketInputs
from .metrics import arithmetic_sharpe, compute_metrics, geometric_excess_sharpe
from .optimizer import SolverError
from .pipeline import (
    StrategyResult,
    run_fixed_weight,
    run_markowitz,
    run_markowitz_with_models,
    run_vol_controlled_fixed_weight,
)
from .risk import CovarianceHistory, FactorRiskHistory, RollingSampleCovariance
from .simulator import BacktestData, BacktestResults, run_backtest

__all__ = [
    "AlphaForecasts",
    "BacktestData",
    "BacktestResults",
    "CovarianceHistory",
    "EwmaAlpha",
    "FactorRiskHistory",
    "InitialPortfolio",
    "MarketInputs",
    "MarkowitzConfig",
    "RidgeAlpha",
    "RollingSampleCovariance",
    "SolverError",
    "StaticBaseline",
    "StrategyResult",
    "arithmetic_sharpe",
    "compute_metrics",
    "geometric_excess_sharpe",
    "run_backtest",
    "run_fixed_weight",
    "run_markowitz",
    "run_markowitz_with_models",
    "run_vol_controlled_fixed_weight",
]
