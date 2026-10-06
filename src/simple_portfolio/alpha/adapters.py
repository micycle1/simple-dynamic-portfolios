"""Forecast container with explicit units, and the model interface."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ..data import check_complete, check_dates, check_labels


@dataclass(frozen=True)
class AlphaForecasts:
    """Expected simple total returns over ``horizon_days`` trading days.

    ``values`` is dates x assets; a row labelled ``t`` is used at the rebalance
    on ``t``. Horizon conversion is linear (non-compounding), matching the
    paper: an annualized forecast is one with ``horizon_days=periods_per_year``
    and a mean daily return one with ``horizon_days=1``.
    """

    values: pd.DataFrame
    horizon_days: float

    def __post_init__(self) -> None:
        """Validate labels and units."""
        if not isinstance(self.values, pd.DataFrame):
            raise TypeError("Forecast values must be a DataFrame (dates x assets).")
        check_dates(self.values.index, "Forecasts")
        check_labels(self.values.columns, "Forecast columns")
        if np.isinf(self.values.to_numpy(dtype=float)).any():
            raise ValueError("Forecasts must not be infinite.")
        if not self.horizon_days > 0:
            raise ValueError("Forecast horizon_days must be positive.")

    @classmethod
    def from_annualized(cls, values: pd.DataFrame, periods_per_year: int = 252) -> "AlphaForecasts":
        """Forecasts stated as annualized (linearly scaled) simple returns."""
        return cls(values, horizon_days=periods_per_year)

    def rescale_to_horizon(self, horizon_days: float) -> "AlphaForecasts":
        """Linearly rescale to another horizon, e.g. the optimizer's."""
        if horizon_days == self.horizon_days:
            return self
        return AlphaForecasts(self.values * (horizon_days / self.horizon_days), horizon_days)

    def aligned(self, dates: pd.DatetimeIndex, assets: pd.Index) -> np.ndarray:
        """Values on ``dates`` in ``assets`` order; every entry must be finite."""
        return check_complete(self.values, dates, assets, "Alpha forecasts").to_numpy(dtype=float)


class AlphaModel(Protocol):
    """Anything that turns prepared inputs into a forecast history."""

    def forecast(
        self, prices: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> AlphaForecasts:
        """Forecast history for every price date."""
        ...
