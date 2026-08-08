"""Shared test fixtures for tests/unit.

Tests that patch config.DUCKDB_PATH to a fresh tmp_path file (to get a
clean, empty database) call storage.get_connection() with no explicit
path - the same "path is None" branch that triggers seed loading (see
storage._load_seed) for the real app. Without this, those tests would
unintentionally load the real, committed data/seed/ snapshot instead of
starting empty. Tests that specifically want to exercise seed loading
override this within their own scope.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def no_seed_by_default(tmp_path):
    with patch("src.data.storage.config.SEED_DIR", tmp_path / "no_seed_here"):
        yield
