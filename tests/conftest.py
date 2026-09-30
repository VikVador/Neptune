r"""Pytest configuration: integration mark and GPFS-aware auto-skip."""

import pytest

from neptune.config import (
    PATH_CONDITIONING,
    PATH_MASK,
    PATH_PATHS,
    PATH_STATS,
    PATH_STATS_CONDITIONING,
    PATH_STATS_INCREMENTS,
)


def pytest_configure(config: pytest.Config) -> None:
    r"""Register the 'integration' marker for tests requiring GPFS paths."""

    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring GPFS paths (skipped in CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    r"""Auto-skip integration tests if GPFS paths are unavailable."""

    paths = (
        PATH_MASK,
        PATH_PATHS,
        PATH_CONDITIONING,
        PATH_STATS_INCREMENTS,
        *PATH_STATS.values(),
        *PATH_STATS_CONDITIONING.values(),
    )
    if all(path.exists() for path in paths):
        return
    skip_integration = pytest.mark.skip(reason="GPFS paths unavailable")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
