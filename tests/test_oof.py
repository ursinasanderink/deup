"""P-min-core: out-of-fold error collection.

Includes the two gates from the plan:
  * parity-exact: collected OOF errors equal a manual fold-by-fold computation;
  * leakage gate: the collector reports honest out-of-sample errors, NOT the
    near-zero in-sample errors a leaky implementation would produce.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor

from deup.core import OOFErrorCollector
from deup.splitters import PurgedWalkForward


def _make_regression(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(scale=0.5, size=n)
    return X, y


def test_oof_parity_exact_with_manual_loop() -> None:
    X, y = _make_regression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    collector = OOFErrorCollector(LinearRegression(), cv=cv, loss="squared")
    result = collector.fit_collect(X, y)

    # Manual fold-by-fold OOF predictions with the SAME splits.
    manual_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(X, y):
        m = clone(LinearRegression())
        m.fit(X[train_idx], y[train_idx])
        manual_pred[test_idx] = m.predict(X[test_idx])
    manual_err = (y - manual_pred) ** 2

    assert result.n == len(y)
    assert np.allclose(result.predictions, manual_pred)
    assert np.allclose(result.errors, manual_err)


def test_leakage_gate_reports_out_of_sample_errors() -> None:
    """A 1-NN base model has ~0 in-sample error but real out-of-sample error.

    If the collector ever leaked (predicting rows the model was trained on), the
    collected errors would collapse toward zero and this assertion would fail.
    """
    X, y = _make_regression(n=300, seed=1)
    knn = KNeighborsRegressor(n_neighbors=1)

    # honest out-of-fold errors
    oof = OOFErrorCollector(knn, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    oof_err = oof.fit_collect(X, y).errors

    # what a leaky (in-sample) implementation would report
    leaky = clone(knn).fit(X, y)
    in_sample_err = (y - leaky.predict(X)) ** 2

    assert in_sample_err.mean() < 1e-9  # 1-NN memorizes the training set
    # honest OOF error must be materially larger than the leaked ~0 error
    assert oof_err.mean() > 100 * (in_sample_err.mean() + 1e-12)
    assert oof_err.mean() > 0.1


def test_refit_on_all_exposes_deployable_estimator() -> None:
    X, y = _make_regression()
    res = OOFErrorCollector(
        LinearRegression(), cv=KFold(n_splits=4), refit_on_all=True
    ).fit_collect(X, y)
    assert res.estimator is not None
    assert res.estimator.predict(X[:3]).shape == (3,)

    res_no = OOFErrorCollector(
        LinearRegression(), cv=KFold(n_splits=4), refit_on_all=False
    ).fit_collect(X, y)
    assert res_no.estimator is None


def test_walk_forward_excludes_unpredicted_early_rows() -> None:
    X, y = _make_regression(n=120)
    res = OOFErrorCollector(
        LinearRegression(), cv=PurgedWalkForward(n_splits=4, embargo=2)
    ).fit_collect(X, y)
    # earliest rows are never in a test fold, so fewer than n rows are returned
    assert res.n < len(y)
    assert res.n > 0


def test_classification_logloss_path() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    res = OOFErrorCollector(
        LogisticRegression(),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        loss="logloss",
        proba=True,
    ).fit_collect(X, y)
    assert np.all(res.errors >= 0)
    assert res.n == len(y)


def test_rank_loss_with_group_coherent_walk_forward() -> None:
    # panel: 10 dates x 8 assets
    n_dates, n_assets = 10, 8
    rng = np.random.default_rng(7)
    dates = np.repeat(np.arange(n_dates), n_assets)
    X = rng.normal(size=(n_dates * n_assets, 3))
    y = X[:, 0] + rng.normal(scale=0.3, size=n_dates * n_assets)
    res = OOFErrorCollector(
        LinearRegression(),
        cv=PurgedWalkForward(n_splits=3, embargo=0),
        loss="rank",
    ).fit_collect(X, y, groups=dates)
    assert res.group_ids is not None
    assert np.all((res.errors >= 0) & (res.errors <= 1))


def test_multiclass_proba_brier_path() -> None:
    """Multiclass predict_proba must not crash and must store 2-D predictions."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    logits = X @ rng.normal(size=(4, 3))
    y = logits.argmax(axis=1)
    res = OOFErrorCollector(
        LogisticRegression(max_iter=500),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        loss="brier",
        proba=True,
    ).fit_collect(X, y)
    assert res.predictions.ndim == 2
    assert res.predictions.shape[1] == 3
    assert np.all(res.errors >= 0)
    assert res.n == len(y)


def test_overlapping_folds_warns_and_keeps_one_error_per_row() -> None:
    X, y = _make_regression(n=120)
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    with pytest.warns(UserWarning, match="multiple test folds"):
        res = OOFErrorCollector(LinearRegression(), cv=cv).fit_collect(X, y)
    # one error per row despite rows being tested in every repeat
    assert res.n == len(y)


def test_groups_length_mismatch_raises() -> None:
    X, y = _make_regression(n=50)
    with pytest.raises(ValueError, match="groups length"):
        OOFErrorCollector(LinearRegression(), cv=KFold(n_splits=3)).fit_collect(
            X, y, groups=np.arange(10)
        )
