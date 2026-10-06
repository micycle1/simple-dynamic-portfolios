"""Exponentially weighted mean of daily returns ("Simple Markowitz")."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .adapters import AlphaForecasts


def ewma_mean_returns(
    prices: pd.DataFrame,
    assets: Sequence[str] | None = None,
    halflife: float = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """EWMA of daily simple returns; zero during warm-up and for assets not listed."""
    columns = list(prices.columns) if assets is None else list(assets)
    return (
        prices[columns]
        .pct_change()
        .ewm(halflife=halflife, min_periods=min_periods)
        .mean()
        .reindex(columns=prices.columns, fill_value=0.0)
        .fillna(0.0)
    )


@dataclass(frozen=True)
class EwmaAlpha:
    """EWMA alpha model; ``forecast`` returns expected daily returns."""

    halflife: float = 252
    min_periods: int = 63
    assets: tuple[str, ...] | None = None

    def forecast(
        self, prices: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> AlphaForecasts:
        """Forecast history for every price date (``features`` is ignored)."""
        values = ewma_mean_returns(prices, self.assets, self.halflife, self.min_periods)
        return AlphaForecasts(values, horizon_days=1)
