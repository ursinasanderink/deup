"""Structural typing for the models ``deup`` wraps.

DEUP is model-agnostic: it orchestrates a *base* predictor and a *secondary* error
predictor that both follow the scikit-learn ``fit`` / ``predict`` convention. We
express that requirement structurally (via :class:`typing.Protocol`) rather than by
inheritance, so any duck-typed estimator — scikit-learn, LightGBM, XGBoost, a thin
wrapper around a neural net — qualifies without importing scikit-learn here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt


@runtime_checkable
class Predictor(Protocol):
    """A minimal scikit-learn-style point predictor.

    Any object exposing ``fit(X, y)`` and ``predict(X)`` satisfies this protocol.
    The return type of ``fit`` is intentionally ``Any`` (scikit-learn returns
    ``self``); only the *presence* of the methods is required.
    """

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Any:
        """Fit the predictor on features ``X`` and targets ``y``."""
        ...

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Return point predictions for ``X``."""
        ...


@runtime_checkable
class ProbabilisticPredictor(Predictor, Protocol):
    """A classifier that additionally exposes ``predict_proba``."""

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Return class-probability estimates for ``X``."""
        ...
