"""Smoke tests for the package scaffold (P0)."""

from __future__ import annotations

import deup


def test_version_is_nonempty_string() -> None:
    assert isinstance(deup.__version__, str)
    assert deup.__version__
