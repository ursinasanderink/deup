"""P-min-est: DEUPRegressor wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from deup import DEUPRegressor
from deup.splitters import PurgedWalkForward


def _make_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(scale=0.5, size=n)
    return X, y


def test_fit_predict_shapes() -> None:
    X, y = _make_data()
    model = DEUPRegressor(base_model=LinearRegression(), cv=4, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(y),)


def test_return_uncertainty_tuple() -> None:
    X, y = _make_data()
    model = DEUPRegressor(base_model=RandomForestRegressor(n_estimators=50), random_state=0)
    model.fit(X, y)
    out = model.predict(X, return_uncertainty=True)
    assert isinstance(out, tuple)
    pred, unc = out
    assert pred.shape == unc.shape == (len(y),)
    assert np.all(unc >= 0.0)


def test_default_base_model_works() -> None:
    X, y = _make_data(n=200)
    model = DEUPRegressor(random_state=0).fit(X, y)  # default HGB base + error model
    pred, unc = model.predict(X[:10], return_uncertainty=True)
    assert pred.shape == (10,)
    assert np.all(unc >= 0.0)


def test_uncertainty_tracks_heteroscedastic_noise() -> None:
    # noise scale grows with |x0|, so true error should be larger there;
    # g(x) should rank test points by realized error better than chance.
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(size=(n, 3))
    noise = rng.normal(size=n) * (0.1 + 2.0 * np.abs(X[:, 0]))
    y = X[:, 1] + noise

    n_tr = 1000
    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=80, random_state=0),
        cv=5,
        random_state=0,
    ).fit(X[:n_tr], y[:n_tr])

    pred, unc = model.predict(X[n_tr:], return_uncertainty=True)
    realized_sq_err = (y[n_tr:] - pred) ** 2
    rho = spearmanr(unc, realized_sq_err).statistic
    assert rho > 0.2


def test_works_with_walk_forward_cv() -> None:
    X, y = _make_data(n=300)
    model = DEUPRegressor(
        base_model=LinearRegression(),
        cv=PurgedWalkForward(n_splits=4, embargo=2),
    ).fit(X, y)
    pred, unc = model.predict(X[:5], return_uncertainty=True)
    assert pred.shape == (5,)
    assert np.all(unc >= 0.0)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        DEUPRegressor().predict(np.zeros((3, 4)))


def test_sklearn_clone_and_params() -> None:
    model = DEUPRegressor(base_model=LinearRegression(), cv=3, log_target=False)
    cloned = clone(model)
    assert cloned.get_params()["cv"] == 3
    assert cloned.get_params()["log_target"] is False
    # base_model is preserved as an (unfitted) estimator instance
    assert isinstance(cloned.get_params()["base_model"], LinearRegression)


def test_log_vs_raw_target_both_nonnegative() -> None:
    X, y = _make_data(n=200)
    for transform in ("log", "asinh", "none"):
        m = DEUPRegressor(
            base_model=LinearRegression(),
            cv=4,
            target_transform=transform,
            random_state=0,
        ).fit(X, y)
        unc = m.predict_epistemic(X)
        assert np.all(unc >= 0.0)
