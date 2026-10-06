"""Untaxed portfolio backtest simulator.

Array-based and positional: callers are responsible for aligning assets and
dates (``simple_dynamic_portfolios.data`` does this for labelled inputs). The daily event
order is, for each observation ``i``:

1. Accrue cash at ``ffrs[i]`` (a daily decimal return).
2. Apply asset returns ``returns[i]`` (from total-return prices).
3. Liquidate holdings of assets that are unavailable on ``i``.
4. Record NAV, net of any liquidation cost.
5. Rebalance if scheduled, or on the first observation.
6. Charge trading costs to cash going forward (they reach the *next* NAV).
7. Record composition (target weights on rebalance days) and turnover.

Compatibility characteristics preserved from the paper implementation:

- The first observation has zero asset and zero cash return.
- A rebalance day's NAV is recorded before that rebalance's costs, so the final
  rebalance's costs never reach a recorded NAV.
- Composition on a rebalance day is the pre-cost target, and a fully invested
  target can leave slightly negative cash after costs.
- A forced liquidation on a day *without* a rebalance reduces that day's
  recorded NAV by its cost, but the cost is not debited from cash, so it does
  not persist. Rebalance days do persist it. Labelled inputs built through
  ``simple_dynamic_portfolios.data`` require a complete price panel, so this path is
  only reachable through ``BacktestData.from_pandas`` directly.
"""

import pickle
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .optimizer import PortfolioConstructor

RebalanceFrequency = Literal["D", "W", "M", "Q", "Y"]


class BacktestResults(NamedTuple):
    """Results of a backtest."""

    navs: pd.Series
    composition: pd.DataFrame
    turnover: pd.Series
    metadata: pd.Series

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestResults":
        """Load results from a dictionary."""
        return cls(
            navs=data["navs"],
            composition=data["composition"],
            turnover=data["turnover"],
            metadata=data["metadata"],
        )

    def to_dict(self) -> dict:
        """Save results to a dictionary."""
        return {
            "navs": self.navs,
            "composition": self.composition,
            "turnover": self.turnover,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save results to a pickle file."""
        with path.open("wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> "BacktestResults":
        """Load results from a pickle file."""
        with path.open("rb") as f:
            return cls.from_dict(pickle.load(f))  # noqa: S301

    def between(self, start=None, end=None) -> "BacktestResults":
        """Restrict every series to the inclusive interval ``[start, end]``."""
        return BacktestResults(
            navs=self.navs.loc[start:end],
            composition=self.composition.loc[start:end],
            turnover=self.turnover.loc[start:end],
            metadata=self.metadata,
        )


def rebalance_schedule(timeline: pd.DatetimeIndex, rebal_freq: RebalanceFrequency) -> np.ndarray:
    """Flag the last observation of each period (including a trailing partial period)."""
    periods = pd.Series(timeline).dt.to_period(rebal_freq)
    return np.array(~periods.duplicated(keep="last"), dtype=bool)


class BacktestData(NamedTuple):
    """Positional arrays consumed by ``run_backtest``."""

    timeline: np.ndarray
    rebal_schedule: np.ndarray
    assets: np.ndarray
    returns: np.ndarray
    universe: np.ndarray
    ffrs: np.ndarray

    @classmethod
    def from_pandas(
        cls,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        closes: pd.DataFrame,
        ffrs: pd.Series,
        rebal_freq: RebalanceFrequency,
    ) -> "BacktestData":
        """Build arrays from total-return prices and daily cash returns (legacy semantics).

        Missing prices mark an asset unavailable on that date; its returns are
        computed from forward- then back-filled prices. ``ffrs`` must share the
        index of ``closes`` over the interval: it is used positionally.
        """
        closes = closes.loc[start_date:end_date]
        ffrs = ffrs.loc[start_date:end_date]

        timeline = closes.index
        rebal_schedule = rebalance_schedule(timeline, rebal_freq)
        universe_np = closes.notna().to_numpy()

        filled_closes = closes.ffill().bfill()
        prev_closes = filled_closes.shift(1)
        filled_returns_np = np.nan_to_num(
            (filled_closes - prev_closes).to_numpy() / prev_closes.to_numpy()
        )

        # Zero cash return on the first day (no carry before the simulation starts).
        ffrs_np = ffrs.to_numpy().flatten()
        ffrs_np[0] = 0.0

        return cls(
            timeline=timeline.to_numpy(),
            rebal_schedule=rebal_schedule,
            assets=closes.columns.to_numpy(),
            universe=universe_np,
            returns=filled_returns_np,
            ffrs=ffrs_np,
        )


def run_backtest(
    backtest_name: str,
    data: BacktestData,
    portfolio_constructor: PortfolioConstructor,
    bid_ask_spread: float = 5e-4,
    initial_weights: np.ndarray | None = None,
) -> BacktestResults:
    """Run a backtest for a given portfolio strategy.

    ``bid_ask_spread`` is the full spread: trading ``|dw|`` of NAV costs
    ``0.5 * bid_ask_spread * |dw|``. ``initial_weights`` are risky weights as
    fractions of the starting NAV (the remainder is cash); ``None`` starts all
    in cash. The first solve sees these weights and pays to trade out of them.
    """
    timeline = data.timeline
    rebal_schedule = data.rebal_schedule
    n_assets = len(data.assets)
    ffr_1p_np = 1.0 + data.ffrs
    filled_returns_1p_np = 1.0 + data.returns
    universes_np = data.universe
    n_time = len(timeline)

    metadata = pd.Series(
        {
            "name": backtest_name,
            "start_date": timeline[0],
            "end_date": timeline[-1],
        }
    )

    navs_np = np.empty(n_time)
    composition_np = np.zeros((n_time, n_assets))
    turnover_np = np.zeros(n_time, dtype=float)

    potential_liquidation_np: np.ndarray = ~np.all(universes_np, axis=1)

    holdings_np = np.zeros(n_assets)
    cash = 1.0
    if initial_weights is not None:
        initial_weights = np.asarray(initial_weights, dtype=float)
        if initial_weights.shape != (n_assets,) or not np.all(np.isfinite(initial_weights)):
            raise ValueError(f"initial_weights must be {n_assets} finite values.")
        holdings_np[:] = initial_weights
        cash = 1.0 - initial_weights.sum()
    weights_np = np.zeros(n_assets)  # reused buffer — never reassigned
    first_solve = True

    for i in range(n_time):
        universe_np = universes_np[i]

        cash *= ffr_1p_np[i]
        holdings_np *= filled_returns_1p_np[i]
        turnover_dollars = 0.0
        if potential_liquidation_np[i]:
            prev_holdings_sum = holdings_np.sum()
            holdings_np *= universe_np
            liq_value = prev_holdings_sum - holdings_np.sum()
            cash += liq_value
            turnover_dollars = liq_value

        nav = cash + holdings_np.sum() - bid_ask_spread * turnover_dollars
        np.divide(holdings_np, nav, out=weights_np)
        navs_np[i] = nav
        turnover = turnover_dollars / nav

        if rebal_schedule[i] or first_solve:
            new_weights = portfolio_constructor(
                ts=timeline[i],
                curr_weights=weights_np,
                universe=data.universe[i],
                ffr=data.ffrs[i],
            )
            first_solve = False
            new_weights = _checked_weights(new_weights, n_assets, timeline[i])

            # Turnover counts traded (noncash) assets only; cash is excluded.
            new_cash_weight = 1.0 - new_weights.sum()
            tcost_base = np.abs(new_weights - weights_np).sum()
            turnover += 0.5 * tcost_base
            new_cash_weight -= 0.5 * bid_ask_spread * tcost_base

            # Costs come out of cash going forward (reflected in the next NAV).
            cash = new_cash_weight * nav
            np.multiply(new_weights, nav, out=holdings_np)
            weights_np[:] = new_weights

        composition_np[i] = weights_np
        turnover_np[i] = turnover

    composition = pd.DataFrame(composition_np, index=timeline, columns=data.assets)
    composition["Cash"] = 1.0 - composition.sum(axis=1)
    navs = pd.Series(navs_np, index=timeline)
    turnover = pd.Series(turnover_np, index=timeline)
    return BacktestResults(navs, composition, turnover, metadata)


def _checked_weights(weights, n_assets: int, ts) -> np.ndarray:
    """Fail clearly on a missing, misshapen, or non-finite constructor output."""
    if weights is None:
        raise RuntimeError(f"Portfolio constructor returned no weights on {pd.Timestamp(ts)}.")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_assets,) or not np.all(np.isfinite(weights)):
        raise RuntimeError(
            f"Portfolio constructor returned invalid weights on {pd.Timestamp(ts)}: {weights}."
        )
    return weights
