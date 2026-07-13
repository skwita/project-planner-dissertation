"""Root conftest — anchors sys.path so ``import main`` / ``import services...`` work
regardless of where pytest is invoked from, and hosts fixtures shared across tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"


@pytest.fixture
def tiny_task_file() -> str:
    """Path to a small, fast, hand-crafted task graph (5 tasks, 3 roles)."""
    return str(FIXTURES_DIR / "tiny_tasks.csv")
