"""Out-of-fold error collection -- the correctness heart of DEUP.

This implements the paper's Algorithm 2 (K-fold pre-fill of the error dataset): for
each fold, fit a *fresh* clone of the base model on the training rows and predict the
held-out rows, so every row receives an **out-of-sample** prediction. The pointwise
loss of those predictions is the training target for the secondary error predictor.

Training the error predictor on in-sample errors instead (e.g. by predicting rows the
base model was trained on) is the canonical DEUP failure mode -- it underestimates
epistemic uncertainty (Lahlou et al., 2023, Sec. 3.2). The leakage test in the test
suite is designed to fail if this collector ever regresses to in-sample behavior.

Refit assumption
----------------
By default the collector also refits the base model ``f`` on *all* training data and
exposes it as ``OOFResult.estimator`` for deployment. The error predictor ``g`` is
therefore trained on the out-of-sample errors of fold models ``f_{-k}`` (each fit on
a strict subset of the data), but deployed alongside the full-data ``f``. This is the
standard DEUP / stacking assumption: ``g`` learns the error of a *slightly smaller*
model than the one it is paired with at inference. In practice fold and full models
are close for reasonable fold counts; for time-series the fold models are genuinely
smaller (expanding window), which is the realistic operating regime and is handled
explicitly by walk-forward splitters. Set ``refit_on_all=False`` to disable.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from deup.core.losses import LossFn, get_loss
from deup.core.types import OOFResult


def _safe_index(X: Any, idx: npt.NDArray[Any]) -> Any:
    """Row-index ``X`` whether it is a numpy array or a pandas object."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


class OOFErrorCollector:
    """Collect a base model's out-of-fold predictions and pointwise errors.

    Parameters
    ----------
    estimator:
        The base model ``f`` (any scikit-learn-style ``fit``/``predict`` object).
        It is cloned per fold; the passed instance is never fitted in place.
    cv:
        A splitter exposing ``split(X, y, groups)`` (e.g. ``KFold``,
        ``TimeSeriesSplit``, or :class:`deup.splitters.PurgedWalkForward`). For
        time-ordered data use a non-shuffling splitter; the collector itself never
        shuffles.
    loss:
        Error-target loss: a registry name (``"squared"``, ``"absolute"``,
        ``"logloss"``, ``"brier"``, ``"pinball"``, ``"rank"``) or a callable
        ``loss(y_true, y_pred, groups)``.
    proba:
        If ``True``, use ``predict_proba`` instead of ``predict`` -- required for
        classification log-loss / Brier targets. Binary probabilities are stored as
        the positive-class column; multiclass probabilities are stored as a 2-D
        array and passed through to the loss.
    refit_on_all:
        If ``True`` (default), also refit a clone of the base model on all data and
        expose it as ``OOFResult.estimator``. See the module docstring for the
        "g trained on errors of a slightly smaller f" assumption this entails.

    Notes
    -----
    Rows never assigned to a test fold (e.g. the earliest rows under walk-forward)
    are excluded from the returned :class:`~deup.core.types.OOFResult`. If a row is
    assigned to more than one test fold (e.g. repeated CV), a warning is raised and
    the last fold's prediction is kept, since averaging would break the
    one-error-per-row contract that ``g`` is trained on.
    """

    def __init__(
        self,
        estimator: Any,
        cv: Any,
        loss: str | LossFn = "squared",
        *,
        proba: bool = False,
        refit_on_all: bool = True,
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.loss = loss
        self.proba = proba
        self.refit_on_all = refit_on_all

    def _predict_fold(self, model: Any, X_test: Any) -> npt.NDArray[Any]:
        """Return fold predictions: 1-D point preds, pos-class probs, or 2-D probs."""
        if not self.proba:
            return np.asarray(model.predict(X_test), dtype=float)
        proba = np.asarray(model.predict_proba(X_test), dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def fit_collect(
        self, X: Any, y: npt.ArrayLike, groups: npt.ArrayLike | None = None
    ) -> OOFResult:
        """Run the out-of-fold loop and return the collected errors.

        Parameters
        ----------
        X, y:
            Training features and targets.
        groups:
            Optional per-row group labels (e.g. dates). Passed to the splitter and to
            group-aware losses such as ``"rank"``.
        """
        y_arr = np.asarray(y)
        n = y_arr.shape[0]
        groups_arr = None if groups is None else np.asarray(groups)
        if groups_arr is not None and groups_arr.shape[0] != n:
            raise ValueError(f"groups length {groups_arr.shape[0]} != n_rows {n}")

        # Accumulate per-fold predictions so we can size the output array correctly
        # for either point predictions (1-D) or multiclass probabilities (2-D).
        fold_results: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]] = []
        pred_width: int | None = None  # None => 1-D output; int => 2-D output
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y_arr, groups_arr)):
            if len(test_idx) == 0:
                continue  # degenerate fold: nothing held out
            model = clone(self.estimator)
            model.fit(_safe_index(X, train_idx), y_arr[train_idx])
            pred = self._predict_fold(model, _safe_index(X, test_idx))
            if pred.ndim == 2:
                pred_width = pred.shape[1]
            fold_results.append((np.asarray(test_idx), pred))
            del fold  # fold index not needed beyond enumerate

        if not fold_results:
            raise ValueError("No out-of-fold predictions were produced by the splitter.")

        if pred_width is None:
            oof_pred: npt.NDArray[Any] = np.full(n, np.nan, dtype=float)
        else:
            oof_pred = np.full((n, pred_width), np.nan, dtype=float)
        fold_ids = np.full(n, -1, dtype=np.intp)
        hit_count = np.zeros(n, dtype=np.intp)

        for fold, (test_idx, pred) in enumerate(fold_results):
            oof_pred[test_idx] = pred
            fold_ids[test_idx] = fold
            hit_count[test_idx] += 1

        if (hit_count > 1).any():
            n_overlap = int((hit_count > 1).sum())
            warnings.warn(
                f"{n_overlap} row(s) were assigned to multiple test folds; "
                "keeping the last prediction. Use a partitioning splitter for "
                "honest one-error-per-row out-of-fold targets.",
                stacklevel=2,
            )

        mask = fold_ids >= 0
        loss_fn = get_loss(self.loss)
        g_groups = None if groups_arr is None else groups_arr[mask]
        errors = np.asarray(loss_fn(y_arr[mask], oof_pred[mask], g_groups), dtype=float)
        if errors.shape[0] != int(mask.sum()):
            raise ValueError(f"loss returned {errors.shape[0]} values for {int(mask.sum())} rows")

        estimator_ = None
        if self.refit_on_all:
            estimator_ = clone(self.estimator)
            estimator_.fit(X, y_arr)

        return OOFResult(
            predictions=oof_pred[mask],
            errors=errors,
            fold_ids=fold_ids[mask],
            group_ids=g_groups,
            indices=np.flatnonzero(mask),
            estimator=estimator_,
        )
