"""Typed, immutable result containers passed between DEUP's layers.

These dataclasses are the contract between the out-of-fold collector, the error
estimator, and the calibration layer. They are frozen (their fields cannot be
reassigned) and validate array-length consistency on construction, so a malformed
result fails loudly at the boundary rather than silently downstream.

``eq=False`` is deliberate: the fields are numpy arrays, whose element-wise ``==``
does not yield a single boolean, so an auto-generated ``__eq__`` would be a footgun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, eq=False)
class OOFResult:
    """Out-of-fold artifacts produced when collecting a base model's errors.

    Attributes
    ----------
    predictions:
        Out-of-fold predictions of the base model ``f``, one per row.
    errors:
        Per-row error targets that the secondary predictor ``g`` will learn from
        (e.g. squared residuals or per-group rank losses).
    fold_ids:
        The fold in which each row was held out. Useful for diagnostics and for
        walk-forward reporting.
    group_ids:
        Optional per-row group label (e.g. a date for cross-sectional ranking).
        ``None`` for i.i.d. data.
    estimator:
        Optionally, the base model refit on all data for deployment. ``None`` if
        the caller chose not to refit.
    """

    predictions: npt.NDArray[Any]
    errors: npt.NDArray[Any]
    fold_ids: npt.NDArray[Any]
    group_ids: npt.NDArray[Any] | None = None
    estimator: Any = field(default=None)

    def __post_init__(self) -> None:
        preds = np.asarray(self.predictions)
        errs = np.asarray(self.errors)
        folds = np.asarray(self.fold_ids)
        object.__setattr__(self, "predictions", preds)
        object.__setattr__(self, "errors", errs)
        object.__setattr__(self, "fold_ids", folds)

        n = preds.shape[0]
        if errs.shape[0] != n or folds.shape[0] != n:
            raise ValueError(
                "OOFResult arrays must share length: "
                f"predictions={preds.shape[0]}, errors={errs.shape[0]}, "
                f"fold_ids={folds.shape[0]}"
            )
        if self.group_ids is not None:
            groups = np.asarray(self.group_ids)
            object.__setattr__(self, "group_ids", groups)
            if groups.shape[0] != n:
                raise ValueError(f"group_ids length {groups.shape[0]} != n_rows {n}")

    @property
    def n(self) -> int:
        """Number of rows."""
        return int(self.predictions.shape[0])


@dataclass(frozen=True, eq=False)
class UncertaintyResult:
    """A prediction together with its uncertainty decomposition.

    Attributes
    ----------
    prediction:
        Point prediction of the base model.
    epistemic:
        Estimated epistemic uncertainty ``g(x)`` (optionally net of aleatoric).
    aleatoric:
        Optional estimated aleatoric (irreducible) uncertainty ``a(x)``.
    lower, upper:
        Optional calibrated prediction-interval bounds.
    """

    prediction: npt.NDArray[Any]
    epistemic: npt.NDArray[Any]
    aleatoric: npt.NDArray[Any] | None = None
    lower: npt.NDArray[Any] | None = None
    upper: npt.NDArray[Any] | None = None

    def __post_init__(self) -> None:
        pred = np.asarray(self.prediction)
        epi = np.asarray(self.epistemic)
        object.__setattr__(self, "prediction", pred)
        object.__setattr__(self, "epistemic", epi)

        n = pred.shape[0]
        if epi.shape[0] != n:
            raise ValueError(f"epistemic length {epi.shape[0]} != prediction length {n}")
        for name in ("aleatoric", "lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value)
                object.__setattr__(self, name, arr)
                if arr.shape[0] != n:
                    raise ValueError(f"{name} length {arr.shape[0]} != prediction length {n}")

    @property
    def n(self) -> int:
        """Number of rows."""
        return int(self.prediction.shape[0])
