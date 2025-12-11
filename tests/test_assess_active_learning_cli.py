"""Smoke tests for the active-learning assessment CLI placeholder."""

from __future__ import annotations

import pytest

import scripts.assess_active_learning as assess


def test_assess_active_learning_cli_is_disabled() -> None:
    with pytest.raises(SystemExit) as excinfo:
        assess.main()

    message = str(excinfo.value)
    assert "disabled" in message.lower()
    assert "Stage1CoeffSummaries" in message
