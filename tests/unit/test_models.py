"""Alpha and risk estimators and their prepared-input adapters."""

import unittest

import numpy as np
import pandas as pd

from simple_dynamic_portfolios import (
    AlphaForecasts,
    CovarianceHistory,
    EwmaAlpha,
    FactorRiskHistory,
    RidgeAlpha,
    RollingSampleCovariance,
)
from simple_dynamic_portfolios.alpha import ridge_forecast_history


def random_market(n_days: int = 400, n_assets: int = 3, seed: int = 0):
    """Random-walk prices and two arbitrary features."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    assets = [f"asset {j}" for j in range(n_assets)]
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(3e-4, 0.01, (n_days, n_assets)), 0),
                          index=dates, columns=assets)
    features = pd.DataFrame({"level": np.cumsum(rng.normal(size=n_days)),
                             "noise ~ weird name": rng.normal(size=n_days)}, index=dates)
    return prices, features


class TestAlphaForecasts(unittest.TestCase):
    """Units and alignment of prepared forecasts."""

    def test_linear_horizon_rescaling(self) -> None:
        """Annualized forecasts scale to 21 days by 21/252, daily ones by 21."""
        values = pd.DataFrame({"a": [0.12]}, index=pd.DatetimeIndex(["2020-01-31"]))
        annual = AlphaForecasts.from_annualized(values)
        self.assertAlmostEqual(annual.rescale_to_horizon(21).values.iloc[0, 0], 0.01)
        daily = AlphaForecasts(values, horizon_days=1)
        self.assertAlmostEqual(daily.rescale_to_horizon(21).values.iloc[0, 0], 2.52)

    def test_alignment_requires_every_decision_date_and_asset(self) -> None:
        """Missing dates or assets fail; extra assets are misalignment, not ignored."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        forecasts = AlphaForecasts(pd.DataFrame({"a": [0.1, np.nan, 0.2], "b": 0.0}, index=dates),
                                   horizon_days=21)
        np.testing.assert_array_equal(forecasts.aligned(dates[[0, 2]], pd.Index(["b", "a"])),
                                      [[0.0, 0.1], [0.0, 0.2]])
        with self.assertRaisesRegex(ValueError, "first gap: a"):
            forecasts.aligned(dates, pd.Index(["a", "b"]))
        with self.assertRaisesRegex(ValueError, "missing assets"):
            forecasts.aligned(dates[:1], pd.Index(["a", "b", "c"]))
        with self.assertRaisesRegex(ValueError, "outside the investable universe"):
            forecasts.aligned(dates[:1], pd.Index(["a"]))


class TestRidge(unittest.TestCase):
    """Walk-forward ridge on arbitrary features."""

    def test_forecasts_use_only_matured_targets(self) -> None:
        """Changing prices after date t leaves forecasts up to t unchanged."""
        prices, features = random_market()
        base = ridge_forecast_history(prices, features, horizon_days=10, min_window=20,
                                      max_history=100)
        cut = 250
        shocked = prices.copy()
        shocked.iloc[cut + 1:] *= 1.5
        after = ridge_forecast_history(shocked, features, horizon_days=10, min_window=20,
                                       max_history=100)
        pd.testing.assert_frame_equal(base.iloc[: cut + 1], after.iloc[: cut + 1])
        self.assertFalse(base.iloc[cut + 11:].equals(after.iloc[cut + 11:]))

    def test_warm_up_is_zero_and_annualized_units(self) -> None:
        """No forecast before min_window matured targets; output is annualized."""
        prices, features = random_market()
        forecasts = RidgeAlpha(horizon_days=10, min_window=20, max_history=100).forecast(
            prices, features)
        self.assertEqual(forecasts.horizon_days, 252)
        values = forecasts.values.to_numpy()
        np.testing.assert_array_equal(values[: 20 + 10 - 1], 0.0)
        self.assertTrue(np.abs(values[20 + 10 - 1]).sum() > 0)

    def test_features_must_share_the_price_index(self) -> None:
        """Features are not silently reindexed."""
        prices, features = random_market()
        with self.assertRaisesRegex(ValueError, "same index"):
            ridge_forecast_history(prices, features.iloc[1:])


class TestEwma(unittest.TestCase):
    """EWMA forecasts are daily and zero during warm-up."""

    def test_units_and_warm_up(self) -> None:
        """Horizon of one day; zeros before min_periods returns."""
        prices, _ = random_market()
        forecasts = EwmaAlpha(halflife=20, min_periods=30).forecast(prices)
        self.assertEqual(forecasts.horizon_days, 1)
        np.testing.assert_array_equal(forecasts.values.iloc[:30].to_numpy(), 0.0)
        expected = prices.pct_change().ewm(halflife=20).mean().iloc[-1]
        pd.testing.assert_series_equal(forecasts.values.iloc[-1], expected)


class TestRisk(unittest.TestCase):
    """Rolling estimator and prepared covariance adapters."""

    def test_rolling_estimator_reconstructs_the_sample_covariance(self) -> None:
        """F F^T equals the centered sample covariance of the trailing window."""
        prices, _ = random_market()
        risk = RollingSampleCovariance(window=11).estimate(prices)
        date = prices.index[200]
        window = prices.pct_change().loc[:date].tail(11)
        np.testing.assert_allclose(risk.covariance(date), window.cov() + np.eye(3) * 1e-14,
                                   rtol=1e-10, atol=1e-18)

    def test_rolling_window_must_exceed_universe_size(self) -> None:
        """The Cholesky estimator is not a default for larger universes."""
        prices, _ = random_market(n_assets=12)
        with self.assertRaisesRegex(ValueError, "must exceed the number of assets"):
            RollingSampleCovariance(window=11).estimate(prices)

    def test_covariance_input_is_validated_and_factored_exactly(self) -> None:
        """Symmetric PSD matrices become an exact factor form with no added residual."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        sigma = np.array([[4.0, 1.0], [1.0, 1.0]]) * 1e-4
        values = pd.concat({d: pd.DataFrame(sigma, index=["a", "b"], columns=["a", "b"])
                            for d in dates}, names=["date", "asset"])
        risk = CovarianceHistory(values).to_factor_risk()
        np.testing.assert_allclose(risk.covariance(dates[1]), sigma, atol=1e-18)
        np.testing.assert_array_equal(risk.residual_std, 0.0)

        asymmetric = values.copy()
        asymmetric.iloc[1, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            CovarianceHistory(asymmetric)
        indefinite = values * np.array([1.0, -1.0])
        with self.assertRaises(ValueError):
            CovarianceHistory(indefinite)

    def test_constructor_dict_aligns_by_label(self) -> None:
        """Risk rows follow the requested asset order, not their stored order."""
        dates = pd.bdate_range("2020-01-01", periods=1)
        loadings = np.array([[[1.0], [2.0]]])
        risk = FactorRiskHistory.from_arrays(dates, pd.Index(["a", "b"]), loadings,
                                             np.array([[0.1, 0.2]]))
        model = risk.to_constructor_dict(dates, pd.Index(["b", "a"]))
        np.testing.assert_array_equal(model["Fs"][dates[0]], [[2.0], [1.0]])
        np.testing.assert_array_equal(model["D_halves"][dates[0]], [0.2, 0.1])
        with self.assertRaisesRegex(ValueError, "missing 1 decision dates"):
            risk.to_constructor_dict(pd.bdate_range("2020-01-02", periods=1), pd.Index(["a", "b"]))


if __name__ == "__main__":
    unittest.main()
