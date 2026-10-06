"""Strategy configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .simulator import RebalanceFrequency

_FREQUENCIES = ("D", "W", "M", "Q", "Y")


@dataclass(frozen=True)
class MarkowitzConfig:
    """Settings for the anchored, volatility-limited Markowitz strategy.

    Units are explicit: ``annual_vol_target`` is annualized with
    ``periods_per_year`` and converted to the daily units of the risk model;
    ``objective_horizon_days`` is the horizon (in trading days) of both the
    optimizer's expected returns and its compounded cash return.

    ``bid_ask_spread`` is the full spread charged by the simulator and, unless
    ``optimizer_bid_ask_spread`` is set, assumed by the optimizer. Setting the
    latter is the only way to make the two differ.
    """

    annual_vol_target: float = 0.07
    max_risky_exposure: float = 1.0
    relative_l1_radius: float = 1.0
    rebalance_frequency: RebalanceFrequency = "M"
    objective_horizon_days: int = 21
    bid_ask_spread: float = 5e-4
    optimizer_bid_ask_spread: float | None = None
    periods_per_year: int = 252
    solver: str = "CLARABEL"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    accept_inaccurate_solutions: bool = False

    def __post_init__(self) -> None:
        """Validate parameter domains."""
        if not self.annual_vol_target > 0.0:
            raise ValueError("annual_vol_target must be positive.")
        if not self.max_risky_exposure > 0.0:
            raise ValueError("max_risky_exposure must be positive.")
        if not self.relative_l1_radius >= 0.0:
            raise ValueError("relative_l1_radius must be nonnegative.")
        if self.rebalance_frequency not in _FREQUENCIES:
            raise ValueError(f"rebalance_frequency must be one of {_FREQUENCIES}.")
        if self.objective_horizon_days <= 0:
            raise ValueError("objective_horizon_days must be positive.")
        if not self.bid_ask_spread >= 0.0:
            raise ValueError("bid_ask_spread must be nonnegative.")
        if self.optimizer_bid_ask_spread is not None and not self.optimizer_bid_ask_spread >= 0.0:
            raise ValueError("optimizer_bid_ask_spread must be nonnegative.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")

    @property
    def daily_vol_target(self) -> float:
        """Volatility limit in the daily units of the risk model."""
        return self.annual_vol_target / np.sqrt(self.periods_per_year)

    @property
    def optimizer_spread(self) -> float:
        """Full spread assumed by the optimizer's trading-cost term."""
        if self.optimizer_bid_ask_spread is None:
            return self.bid_ask_spread
        return self.optimizer_bid_ask_spread
