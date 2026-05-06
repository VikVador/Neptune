r"""Pytest configuration: integration mark and GPFS-aware auto-skip."""

import pytest

from neptune.config import PATH_MASK, PATH_PATHS, PATH_STATS


def pytest_configure(config: pytest.Config) -> None:
    """Register the 'integration' marker for tests requiring GPFS paths."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring GPFS paths (skipped in CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip integration tests if GPFS paths are unavailable."""
    if PATH_MASK.exists() and PATH_STATS.exists() and PATH_PATHS.exists():
        return
    skip_integration = pytest.mark.skip(reason="GPFS paths unavailable")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
