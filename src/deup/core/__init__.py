"""Core protocols, typed result containers, and the grouped/panel data model.

These are the framework-agnostic foundations every estimator in ``deup`` is built
on. Nothing here imports a heavy backend (no torch, lightgbm, or pandas); the only
runtime dependency is numpy.
"""

from __future__ import annotations

from deup.core.grouping import Grouping
from deup.core.protocols import Predictor, ProbabilisticPredictor
from deup.core.types import OOFResult, UncertaintyResult

__all__ = [
    "Grouping",
    "OOFResult",
    "Predictor",
    "ProbabilisticPredictor",
    "UncertaintyResult",
]
