"""Pytest configuration and shared fixtures for TrustSync tests."""
from pathlib import Path
import pytest
from trustsync.config import Config, load_config


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command-line option --runslow."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip items marked slow unless --runslow is supplied."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def spec_cfg() -> Config:
    """Load default specification config fixture."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "config_spec_defaults.yaml"
    return load_config(fixture_path)

