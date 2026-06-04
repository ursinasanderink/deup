"""P1: result containers validate and are immutable."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from deup.core import OOFResult, UncertaintyResult


def test_oof_result_basic() -> None:
    r = OOFResult(
        predictions=np.arange(5.0),
        errors=np.ones(5),
        fold_ids=np.zeros(5, dtype=int),
    )
    assert r.n == 5
    assert r.group_ids is None


def test_oof_result_with_groups() -> None:
    r = OOFResult(
        predictions=np.arange(4.0),
        errors=np.ones(4),
        fold_ids=np.zeros(4, dtype=int),
        group_ids=np.array([0, 0, 1, 1]),
    )
    assert r.group_ids is not None
    assert r.n == 4


def test_oof_result_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share length"):
        OOFResult(
            predictions=np.arange(5.0),
            errors=np.ones(4),
            fold_ids=np.zeros(5, dtype=int),
        )


def test_oof_result_group_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="group_ids length"):
        OOFResult(
            predictions=np.arange(3.0),
            errors=np.ones(3),
            fold_ids=np.zeros(3, dtype=int),
            group_ids=np.array([0, 1]),
        )


def test_oof_result_is_frozen() -> None:
    r = OOFResult(
        predictions=np.arange(3.0),
        errors=np.ones(3),
        fold_ids=np.zeros(3, dtype=int),
    )
    with pytest.raises(FrozenInstanceError):
        r.predictions = np.zeros(3)  # type: ignore[misc]


def test_uncertainty_result_optional_fields() -> None:
    u = UncertaintyResult(prediction=np.zeros(3), epistemic=np.ones(3))
    assert u.aleatoric is None
    assert u.lower is None
    assert u.n == 3


def test_uncertainty_result_validates_length() -> None:
    with pytest.raises(ValueError, match="epistemic length"):
        UncertaintyResult(prediction=np.zeros(3), epistemic=np.ones(2))


def test_uncertainty_result_validates_interval_length() -> None:
    with pytest.raises(ValueError, match="lower length"):
        UncertaintyResult(
            prediction=np.zeros(3),
            epistemic=np.ones(3),
            lower=np.zeros(2),
        )
