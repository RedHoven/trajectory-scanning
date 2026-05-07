"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def out_path(tmp_path):
    """Provide a temporary output directory for tests."""
    return tmp_path
