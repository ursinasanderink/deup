"""P1: the model protocols accept duck-typed estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from deup.core import Predictor, ProbabilisticPredictor


class DummyRegressor:
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> DummyRegressor:
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        return np.zeros(np.asarray(X).shape[0])


class DummyClassifier(DummyRegressor):
    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        n = np.asarray(X).shape[0]
        return np.full((n, 2), 0.5)


def test_regressor_satisfies_predictor() -> None:
    assert isinstance(DummyRegressor(), Predictor)


def test_regressor_is_not_probabilistic() -> None:
    assert not isinstance(DummyRegressor(), ProbabilisticPredictor)


def test_classifier_is_probabilistic() -> None:
    assert isinstance(DummyClassifier(), ProbabilisticPredictor)


def test_arbitrary_object_is_not_a_predictor() -> None:
    obj: Any = object()
    assert not isinstance(obj, Predictor)
