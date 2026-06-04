"""Error-target losses for DEUP.

In DEUP the *error target* for a row is the base model's pointwise loss
``l(y, f(x))`` (Lahlou et al., 2023, Eq. 9 / Alg. 1). The secondary predictor ``g``
then regresses these targets. This module provides the common choices behind a
single ``get_loss`` factory, plus a ``callable`` escape hatch for custom losses.

Each loss has the signature ``loss(y_true, y_pred, groups=None) -> ndarray`` and
returns one non-negative error per row. ``groups`` is only used by group-aware losses
such as :func:`rank_loss`.

**Task pairing**

- regression: ``squared``, ``absolute``, ``pinball``
- classification: ``logloss``, ``brier``
- ranking: ``rank`` (requires ``groups``)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from deup.core.grouping import Grouping

LossFn = Callable[..., npt.NDArray[Any]]
TargetTransform = Literal["none", "log", "asinh"]


def squared_error(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Squared residual ``(y - f(x))**2`` (regression)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: npt.NDArray[Any] = (yt - yp) ** 2
    return out


def absolute_error(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Absolute residual ``|y - f(x)|`` (regression)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: npt.NDArray[Any] = np.abs(yt - yp)
    return out


def log_loss(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Pointwise cross-entropy (classification).

    ``y_pred`` may be a 1-D vector of positive-class probabilities (binary) or a 2-D
    array of class probabilities (multiclass); ``y_true`` holds class indices
    (or 0/1 for binary).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    if yp.ndim == 2:
        probs = np.clip(yp, eps, 1.0)
        true_idx = yt.astype(int)
        chosen = probs[np.arange(probs.shape[0]), true_idx]
        multiclass: npt.NDArray[Any] = -np.log(chosen)
        return multiclass
    p = np.clip(yp, eps, 1.0 - eps)
    yt_f = yt.astype(float)
    binary: npt.NDArray[Any] = -(yt_f * np.log(p) + (1.0 - yt_f) * np.log(1.0 - p))
    return binary


def brier_loss(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Brier score (classification calibration loss).

    Binary: ``(y - p)²`` with ``y ∈ {0,1}``. Multiclass: ``Σ_k (𝟙[y=k] - p_k)²``.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    if yp.ndim == 2:
        n, k = yp.shape
        one_hot = np.zeros((n, k), dtype=float)
        one_hot[np.arange(n), yt.astype(int)] = 1.0
        out: npt.NDArray[Any] = np.sum((one_hot - yp) ** 2, axis=1)
        return out
    yt_f = yt.astype(float)
    out = (yt_f - yp) ** 2
    return out


def pinball_loss(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    *,
    q: float = 0.5,
) -> npt.NDArray[Any]:
    """Pinball / quantile loss for quantile regression (``q ∈ (0, 1)``).

    ``y_pred`` must be the predicted ``q``-quantile of ``y``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"pinball quantile q must be in (0, 1), got {q}")
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    out: npt.NDArray[Any] = np.maximum(q * diff, (q - 1.0) * diff)
    return out


def rank_loss(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Per-group absolute rank displacement (cross-sectional ranking).

    For each group (e.g. a date), ranks ``y_true`` and ``y_pred`` to percentiles and
    returns ``|rank_pct(y_true) - rank_pct(y_pred)|``. Requires a group-coherent
    splitter so that each group's full cross-section is scored together.
    """
    yt = np.asarray(y_true, dtype=float)
    grouping = Grouping.from_labels(groups, n=yt.shape[0])
    rank_true = grouping.rank_within(yt, pct=True)
    rank_pred = grouping.rank_within(y_pred, pct=True)
    out: npt.NDArray[Any] = np.abs(rank_true - rank_pred)
    return out


def apply_error_transform(
    errors: npt.ArrayLike,
    method: TargetTransform = "log",
    *,
    eps: float = 1e-6,
) -> npt.NDArray[Any]:
    """Stabilize heavy-tailed error targets before training ``g``.

    - ``log``: ``log(error + eps)`` (default; used by :class:`~deup.estimators.DEUPRegressor`)
    - ``asinh``: ``asinh(error / eps)`` — robust alternative for very heavy tails
    - ``none``: identity
    """
    err = np.asarray(errors, dtype=float)
    if method == "none":
        return err
    if method == "log":
        out: npt.NDArray[Any] = np.log(err + eps)
        return out
    if method == "asinh":
        out = np.arcsinh(err / eps)
        return out
    raise ValueError(f"Unknown target transform {method!r}. Choose from log, asinh, none.")


def inverse_error_transform(
    values: npt.ArrayLike,
    method: TargetTransform = "log",
    *,
    eps: float = 1e-6,
) -> npt.NDArray[Any]:
    """Map ``g``'s prediction back to the error scale."""
    vals = np.asarray(values, dtype=float)
    if method == "none":
        return vals
    if method == "log":
        out: npt.NDArray[Any] = np.exp(vals) - eps
        return out
    if method == "asinh":
        out = np.sinh(vals) * eps
        return out
    raise ValueError(f"Unknown target transform {method!r}. Choose from log, asinh, none.")


_REGISTRY: dict[str, LossFn] = {
    "squared": squared_error,
    "absolute": absolute_error,
    "logloss": log_loss,
    "brier": brier_loss,
    "rank": rank_loss,
}


def get_loss(loss: str | LossFn, **kwargs: Any) -> LossFn:
    """Resolve ``loss`` (a registry name or a callable) to a loss function.

    For ``pinball``, pass ``q`` (default 0.5) or use the string ``"pinball:0.9"``.
    """
    if callable(loss):
        return loss
    if loss.startswith("pinball"):
        q = float(kwargs.get("q", 0.5))
        if ":" in loss:
            q = float(loss.split(":", 1)[1])
        q_val = q

        def _pinball(
            y_true: npt.ArrayLike,
            y_pred: npt.ArrayLike,
            groups: npt.ArrayLike | None = None,
        ) -> npt.NDArray[Any]:
            return pinball_loss(y_true, y_pred, groups, q=q_val)

        return _pinball
    try:
        return _REGISTRY[loss]
    except KeyError:
        raise ValueError(
            f"Unknown loss {loss!r}. Choose from {sorted(_REGISTRY)} "
            f"(or pinball[:q]) or pass a callable."
        ) from None
