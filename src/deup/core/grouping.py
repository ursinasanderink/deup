"""The grouped / panel data model.

Many DEUP use cases are not i.i.d. row collections: a cross-sectional ranker scores
many assets *per date*, where the loss (rank loss) and any rank-geometry
residualization are defined *within* each date's cross-section. :class:`Grouping`
makes that ``group_by`` concept first-class, while still handling the i.i.d. case
(``group_by=None``) as a single trivial group.

Pure numpy — no pandas dependency in the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def _average_rank(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return 1-based ranks with ties resolved by averaging (scipy 'average').

    Implemented from unique values + counts, so it is independent of input order
    and matches ``pandas.Series.rank(method="average")``.
    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    end = np.cumsum(counts).astype(float)
    start = end - counts + 1.0
    average = (start + end) / 2.0
    result: npt.NDArray[Any] = average[inverse]
    return result


@dataclass(frozen=True)
class Grouping:
    """Maps each row to a group and supports within-group operations.

    Attributes
    ----------
    codes:
        Integer group code per row, in ``[0, n_groups)``.
    labels:
        The unique group labels, indexed by code.
    """

    codes: npt.NDArray[Any]
    labels: npt.NDArray[Any]

    @classmethod
    def from_labels(cls, group_labels: npt.ArrayLike | None, n: int) -> Grouping:
        """Build a grouping from per-row labels.

        Parameters
        ----------
        group_labels:
            Per-row group labels (e.g. dates). If ``None``, all ``n`` rows form a
            single trivial group (the i.i.d. case).
        n:
            Number of rows (used to size the trivial group and validate lengths).
        """
        if group_labels is None:
            return cls(
                codes=np.zeros(n, dtype=np.intp),
                labels=np.zeros(1, dtype=np.intp),
            )
        arr = np.asarray(group_labels)
        if arr.shape[0] != n:
            raise ValueError(f"group_labels length {arr.shape[0]} != n {n}")
        labels, codes = np.unique(arr, return_inverse=True)
        return cls(codes=np.ravel(codes).astype(np.intp), labels=labels)

    @property
    def n_groups(self) -> int:
        """Number of distinct groups."""
        return int(self.labels.shape[0])

    @property
    def is_trivial(self) -> bool:
        """True when there is a single group (the i.i.d. case)."""
        return self.n_groups == 1

    def indices(self) -> list[npt.NDArray[Any]]:
        """Row indices for each group, ordered by group code."""
        return [np.flatnonzero(self.codes == code) for code in range(self.n_groups)]

    def rank_within(self, values: npt.ArrayLike, pct: bool = True) -> npt.NDArray[Any]:
        """Rank ``values`` within each group.

        Ties are averaged. With ``pct=True`` (default) ranks are divided by the
        group size, matching ``pandas.Series.groupby(...).rank(pct=True)`` — the
        convention used for cross-sectional rank features and rank losses.
        """
        vals = np.asarray(values, dtype=float)
        if vals.shape[0] != self.codes.shape[0]:
            raise ValueError(f"values length {vals.shape[0]} != n_rows {self.codes.shape[0]}")
        out = np.empty(vals.shape[0], dtype=float)
        for idx in self.indices():
            ranks = _average_rank(vals[idx])
            if pct:
                ranks = ranks / ranks.shape[0]
            out[idx] = ranks
        return out
