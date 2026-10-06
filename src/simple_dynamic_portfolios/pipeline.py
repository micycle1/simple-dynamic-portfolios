"""Resolve prepared inputs, align them once, run the strategy, and evaluate it.

The optimizer and simulator only ever see arrays aligned here, on the market's
asset order and the simulator's decision dates.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alpha.adapters import AlphaForecasts, AlphaModel
from .baseline import InitialPortfolio, StaticBaseline
from .config import MarkowitzConfig
from .data import MarketInputs, decision_dates
from .metrics import compute_metrics
from .optimizer import (
    AnchoredVolControlPortfolioConstructor,
    FixedWeightPortfolioConstructor,
    FixedWeightVolControlPortfolioConstructor,
)
from .risk import (
    CovarianceHistory,
    FactorRiskHistory,
    RiskModel,
    as_factor_risk,
    rolling_portfolio_volatility,
)
from .simulator import BacktestData, BacktestResults, RebalanceFrequency, run_backtest


@dataclass(frozen=True)
class StrategyResult:
    """Backtest plus per-decision records and the model inputs actually used.

    ``decisions`` holds the target risky weights on each decision date;
    ``diagnostics`` holds per-decision solver status, cash return, and
    exposure. ``alpha`` is in optimizer units (``objective_horizon_days``).
    """

    backtest: BacktestResults
    decisions: pd.DataFrame
    diagnostics: pd.DataFrame
    alpha: AlphaForecasts | None = None
    risk: FactorRiskHistory | None = None

    def evaluate(self, cash_reference: pd.Series | None = None,
                 inflation: pd.Series | None = None, start=None, end=None,
                 **kwargs) -> pd.Series:
        """Metrics over ``[start, end]``; see ``compute_metrics``."""
        return compute_metrics(self.backtest.between(start, end), cash_reference, inflation,
                               **kwargs)


class _Recorder:
    """Log each decision a constructor makes, without changing it."""

    def __init__(self, constructor: Callable, assets: pd.Index) -> None:
        self.constructor = constructor
        self.assets = assets
        self.rows: list[tuple] = []

    def __call__(self, ts, curr_weights, universe, ffr) -> np.ndarray:
        weights = self.constructor(ts=ts, curr_weights=curr_weights, universe=universe, ffr=ffr)
        problem = getattr(self.constructor, "problem", None)
        status = "" if problem is None else problem.status
        self.rows.append((pd.Timestamp(ts), np.array(weights, dtype=float), status, float(ffr)))
        return weights

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        index = pd.DatetimeIndex([row[0] for row in self.rows], name="date")
        decisions = pd.DataFrame([row[1] for row in self.rows], index=index, columns=self.assets)
        diagnostics = pd.DataFrame(
            {
                "status": [row[2] for row in self.rows],
                "cash_return": [row[3] for row in self.rows],
                "risky_exposure": decisions.sum(axis=1),
            },
            index=index,
        )
        return decisions, diagnostics


def _simulate(name: str, data: BacktestData, constructor: Callable, assets: pd.Index,
              bid_ask_spread: float, initial_portfolio: InitialPortfolio | None,
              **artifacts) -> StrategyResult:
    recorder = _Recorder(constructor, assets)
    initial = None if initial_portfolio is None else initial_portfolio.aligned(assets)
    backtest = run_backtest(name, data, recorder, bid_ask_spread=bid_ask_spread,
                            initial_weights=initial)
    decisions, diagnostics = recorder.frames()
    return StrategyResult(backtest, decisions, diagnostics, **artifacts)


def _weights(weights: "StaticBaseline | pd.Series", assets: pd.Index) -> np.ndarray:
    if isinstance(weights, StaticBaseline):
        return weights.aligned(assets)
    return StaticBaseline(weights).aligned(assets)


def run_markowitz(
    market: MarketInputs,
    baseline: StaticBaseline,
    alpha: AlphaForecasts,
    risk: FactorRiskHistory | CovarianceHistory,
    config: MarkowitzConfig | None = None,
    initial_portfolio: InitialPortfolio | None = None,
    start=None,
    end=None,
    name: str = "markowitz",
) -> StrategyResult:
    """Backtest the anchored, volatility-limited Markowitz strategy on prepared inputs.

    ``alpha`` and ``risk`` need finite values on every decision date (the
    first simulation date and each period end); they are used as labelled,
    without lagging. The investable universe is ``market.assets``.
    """
    config = MarkowitzConfig() if config is None else config
    data = market.to_backtest_data(config.rebalance_frequency, start, end)
    assets = market.assets
    decisions = decision_dates(data)
    rows = np.flatnonzero(np.isin(data.timeline, decisions.to_numpy()))
    keys = list(data.timeline[rows])

    forecasts = alpha.rescale_to_horizon(config.objective_horizon_days)
    alphas = np.full((len(data.timeline), len(assets)), np.nan)
    alphas[rows] = forecasts.aligned(decisions, assets)
    factor_risk = as_factor_risk(risk)
    risk_model = factor_risk.to_constructor_dict(decisions, assets, keys=keys)

    constructor = AnchoredVolControlPortfolioConstructor(
        ts_lookup={ts: i for i, ts in enumerate(data.timeline)},
        alphas=alphas,
        risk_model=risk_model,
        vol_target=config.daily_vol_target,
        anchor=baseline.aligned(assets),
        leverage=config.max_risky_exposure,
        bid_ask_spread=config.optimizer_spread,
        cash_rate_horizon_days=config.objective_horizon_days,
        relative_l1_radius=config.relative_l1_radius,
        solver=config.solver,
        solver_options=config.solver_options,
        accept_inaccurate=config.accept_inaccurate_solutions,
    )
    return _simulate(name, data, constructor, assets, config.bid_ask_spread, initial_portfolio,
                     alpha=forecasts, risk=factor_risk)


def run_markowitz_with_models(
    market: MarketInputs,
    baseline: StaticBaseline,
    alpha_model: AlphaModel,
    risk_model: RiskModel,
    features: pd.DataFrame | None = None,
    config: MarkowitzConfig | None = None,
    initial_portfolio: InitialPortfolio | None = None,
    start=None,
    end=None,
    name: str = "markowitz",
) -> StrategyResult:
    """Generate forecast and risk histories from models, then ``run_markowitz``."""
    prices = market.total_return_prices
    return run_markowitz(
        market,
        baseline,
        alpha_model.forecast(prices, features),
        risk_model.estimate(prices),
        config=config,
        initial_portfolio=initial_portfolio,
        start=start,
        end=end,
        name=name,
    )


def run_fixed_weight(
    market: MarketInputs,
    weights: StaticBaseline | pd.Series,
    rebalance_frequency: RebalanceFrequency = "Y",
    leverage: float = 1.0,
    bid_ask_spread: float = 5e-4,
    initial_portfolio: InitialPortfolio | None = None,
    start=None,
    end=None,
    name: str = "fixed_weight",
) -> StrategyResult:
    """Periodically rebalanced constant mix scaled to ``leverage`` risky exposure."""
    data = market.to_backtest_data(rebalance_frequency, start, end)
    assets = market.assets
    fixed = np.tile(_weights(weights, assets), (len(data.timeline), 1))
    constructor = FixedWeightPortfolioConstructor(
        ts_lookup={ts: i for i, ts in enumerate(data.timeline)},
        fixed_weights=fixed,
        leverage=leverage,
    )
    return _simulate(name, data, constructor, assets, bid_ask_spread, initial_portfolio)


def run_vol_controlled_fixed_weight(
    market: MarketInputs,
    weights: StaticBaseline | pd.Series,
    annual_vol_target: float = 0.07,
    vol_window: int = 11,
    max_risky_exposure: float = 1.0,
    rebalance_frequency: RebalanceFrequency = "M",
    bid_ask_spread: float = 5e-4,
    periods_per_year: int = 252,
    volatility: pd.Series | None = None,
    initial_portfolio: InitialPortfolio | None = None,
    start=None,
    end=None,
    name: str = "fixed_weight_vc",
) -> StrategyResult:
    """Constant mix scaled to a volatility target, capped at ``max_risky_exposure``.

    ``volatility`` is the mix's daily volatility estimate by date; by default
    the rolling ``vol_window``-day std of the mix's daily returns.
    """
    data = market.to_backtest_data(rebalance_frequency, start, end)
    assets = market.assets
    mix = _weights(weights, assets)
    if volatility is None:
        volatility = rolling_portfolio_volatility(market.total_return_prices, mix, vol_window)
    decisions = decision_dates(data)
    missing = volatility.reindex(decisions).isna()
    if missing.any():
        raise ValueError(
            f"Volatility estimate is missing on {int(missing.sum())} decision dates, "
            f"first {decisions[missing.to_numpy()][0].date()}; extend the price history."
        )
    constructor = FixedWeightVolControlPortfolioConstructor(
        ts_lookup={ts: i for i, ts in enumerate(data.timeline)},
        vols=volatility.reindex(pd.DatetimeIndex(data.timeline)).to_numpy(),
        fixed_weights=np.tile(mix, (len(data.timeline), 1)),
        vol_target=annual_vol_target / np.sqrt(periods_per_year),
        leverage=max_risky_exposure,
    )
    return _simulate(name, data, constructor, assets, bid_ask_spread, initial_portfolio)


__all__ = [
    "StrategyResult",
    "run_fixed_weight",
    "run_markowitz",
    "run_markowitz_with_models",
    "run_vol_controlled_fixed_weight",
]
