"""P1: the grouped/panel data model (i.i.d. and cross-sectional)."""

from __future__ import annotations

import numpy as np

from deup.core import Grouping


def test_iid_grouping_is_trivial() -> None:
    g = Grouping.from_labels(None, n=5)
    assert g.n_groups == 1
    assert g.is_trivial
    idx = g.indices()
    assert len(idx) == 1
    assert idx[0].tolist() == [0, 1, 2, 3, 4]


def test_panel_grouping_codes() -> None:
    dates = np.array(["d1", "d1", "d2", "d2", "d2"])
    g = Grouping.from_labels(dates, n=5)
    assert g.n_groups == 2
    assert not g.is_trivial
    sizes = sorted(len(i) for i in g.indices())
    assert sizes == [2, 3]


def test_from_labels_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="group_labels length"):
        Grouping.from_labels(np.array(["a", "b"]), n=3)


def test_rank_within_per_group_pct() -> None:
    dates = np.array(["d1", "d1", "d2", "d2", "d2"])
    g = Grouping.from_labels(dates, n=5)
    vals = np.array([10.0, 20.0, 5.0, 15.0, 25.0])
    r = g.rank_within(vals, pct=True)
    # d1: 10 < 20 -> 0.5, 1.0 ; d2: 5 < 15 < 25 -> 1/3, 2/3, 1.0
    assert np.allclose(r[:2], [0.5, 1.0])
    assert np.allclose(r[2:], [1 / 3, 2 / 3, 1.0])


def test_rank_within_averages_ties() -> None:
    g = Grouping.from_labels(np.array(["d", "d", "d", "d"]), n=4)
    vals = np.array([1.0, 1.0, 2.0, 3.0])
    r = g.rank_within(vals, pct=False)
    # tie at value 1 -> average rank (1+2)/2 = 1.5 ; then 3, 4
    assert np.allclose(r, [1.5, 1.5, 3.0, 4.0])


def test_rank_within_length_mismatch_raises() -> None:
    import pytest

    g = Grouping.from_labels(None, n=3)
    with pytest.raises(ValueError, match="values length"):
        g.rank_within(np.array([1.0, 2.0]))
