"""The committed rating_golden.json must stay current.

Regenerate the fixture into a temp path and assert byte-equality with the checked-in file. This
turns the golden from "JS == possibly-stale Python" into a live lock: any change to arena/rating.py
(recompute(), leaderboard(), the tunables) made without re-running gen_rating_golden.py fails here,
which in turn means rating_parity / leaderboard_parity are testing against the *current* Python.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "rating_golden.json"
GEN = ROOT / "tests" / "fixtures" / "gen_rating_golden.py"


def test_rating_golden_is_current(tmp_path):
    out = tmp_path / "regenerated.json"
    # gen_rating_golden.py is hermetic: it pops DATABASE_URL and points ARENA_STORE_DIR at its own
    # temp dir, so this never touches a real store. PYTHONPATH so the standalone script finds `arena`.
    subprocess.run(
        [sys.executable, str(GEN), str(out)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
    )
    assert out.read_text() == FIXTURE.read_text(), (
        "tests/fixtures/rating_golden.json is STALE — regenerate it:\n"
        "  PYTHONPATH=. python tests/fixtures/gen_rating_golden.py tests/fixtures/rating_golden.json"
    )
