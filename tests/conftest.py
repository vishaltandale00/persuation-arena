from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_tests_to_sqlite(monkeypatch):
    """A developer .env may contain DATABASE_URL for Neon.

    Unit tests should not hit that database unless the test explicitly opts into the Postgres path.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
