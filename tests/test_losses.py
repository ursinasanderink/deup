"""P3: error-target losses (full registry + transforms)."""

from __future__ import annotations

import numpy as np
import pytest

from deup.core import (
    apply_error_transform,
    get_loss,
    inverse_error_transform,
)


def test_squared_error() -> None:
    fn = get_loss("squared")
    err = fn(np.array([1.0, 2.0]), np.array([1.5, 0.0]))
    assert np.allclose(err, [0.25, 4.0])


def test_absolute_error() -> None:
    fn = get_loss("absolute")
    err = fn(np.array([1.0, 2.0]), np.array([1.5, 0.0]))
    assert np.allclose(err, [0.5, 2.0])


def test_logloss_binary_is_nonnegative() -> None:
    fn = get_loss("logloss")
    err = fn(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.4]))
    assert np.all(err >= 0)
    assert err[0] < err[2]


def test_logloss_multiclass() -> None:
    fn = get_loss("logloss")
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    err = fn(np.array([0, 2]), probs)
    assert np.allclose(err, -np.log([0.7, 0.8]))


def test_brier_binary() -> None:
    fn = get_loss("brier")
    err = fn(np.array([1.0, 0.0]), np.array([0.8, 0.2]))
    assert np.allclose(err, [0.04, 0.04])


def test_brier_multiclass() -> None:
    fn = get_loss("brier")
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    err = fn(np.array([0, 2]), probs)
    assert np.allclose(err, [0.14, 0.06])


def test_pinball_median_is_half_absolute_error() -> None:
    fn = get_loss("pinball", q=0.5)
    err = fn(np.array([1.0, 2.0]), np.array([1.5, 0.0]))
    assert np.allclose(err, [0.25, 1.0])


def test_pinball_string_quantile() -> None:
    fn = get_loss("pinball:0.9")
    yt = np.array([10.0, 10.0])
    yp = np.array([8.0, 12.0])
    err = fn(yt, yp)
    # under-prediction penalized more at q=0.9
    assert err[0] > err[1]


def test_pinball_invalid_q_raises() -> None:
    with pytest.raises(ValueError, match="pinball quantile"):
        get_loss("pinball", q=1.5)(np.array([1.0]), np.array([1.0]))


def test_rank_loss_perfect_ranking_is_zero() -> None:
    fn = get_loss("rank")
    groups = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 9.0, 8.0, 7.0])
    pred = y.copy()
    err = fn(y, pred, groups)
    assert np.allclose(err, 0.0)


def test_log_transform_roundtrip() -> None:
    err = np.array([0.0, 1.0, 100.0])
    t = apply_error_transform(err, "log", eps=1e-6)
    back = inverse_error_transform(t, "log", eps=1e-6)
    assert np.allclose(back, err, rtol=1e-5)


def test_asinh_transform_roundtrip() -> None:
    err = np.array([0.0, 1.0, 1e4])
    t = apply_error_transform(err, "asinh", eps=1.0)
    back = inverse_error_transform(t, "asinh", eps=1.0)
    assert np.allclose(back, err, rtol=1e-5)


def test_callable_escape_hatch() -> None:
    fn = get_loss(lambda yt, yp, groups=None: np.abs(np.asarray(yt) - np.asarray(yp)))
    assert np.allclose(fn(np.array([3.0]), np.array([1.0])), [2.0])


def test_unknown_loss_raises() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("not-a-loss")
