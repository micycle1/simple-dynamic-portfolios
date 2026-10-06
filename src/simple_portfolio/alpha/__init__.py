"""Return forecasts."""

from .adapters import AlphaForecasts, AlphaModel
from .ridge import RidgeAlpha, fit_ridge_path, ridge_forecast_history
from .simple import EwmaAlpha, ewma_mean_returns

__all__ = [
    "AlphaForecasts",
    "AlphaModel",
    "EwmaAlpha",
    "RidgeAlpha",
    "ewma_mean_returns",
    "fit_ridge_path",
    "ridge_forecast_history",
]
