"""ITEM GATE: arena/statuses.json is the SINGLE SOURCE OF TRUTH for the named RUN/SIGNUP status sets.

Asserts that:
  1. the Python constants in arena.store equal the JSON sets (so store.py reads, not retypes, them);
  2. api/_db.js reads the SAME checked-in file (verified by source inspection — it fs-reads
     ../arena/statuses.json — so the JS lobby and the Python coordinator cannot drift);
  3. the rendered SQL IN(...) fragments preserve the JSON order with no spaces (other queries and the
     JS parity tests match these as exact substrings, so the format is load-bearing).

OUT OF SCOPE (separate L-effort follow-up, deliberately not attempted here): generating the
DDL/CREATE-TABLE column lists from a manifest. This test only covers the status SETS.
"""
import json
from pathlib import Path

from arena import store

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUSES_PATH = REPO_ROOT / "arena" / "statuses.json"
DB_JS_PATH = REPO_ROOT / "api" / "_db.js"


def _load_json_sets() -> dict[str, list[str]]:
    raw = json.loads(STATUSES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def test_statuses_json_is_well_formed():
    sets = _load_json_sets()
    # Every named set is a non-empty list of unique non-empty strings.
    assert sets, "statuses.json defines no status sets"
    for name, members in sets.items():
        assert isinstance(members, list) and members, f"{name} must be a non-empty list"
        assert all(isinstance(m, str) and m for m in members), f"{name} has a non-string/empty member"
        assert len(members) == len(set(members)), f"{name} has duplicate members"


def test_python_constants_equal_json():
    """The store.py constants are LOADED from statuses.json, not retyped — so they must equal it."""
    sets = _load_json_sets()
    assert store.OPEN_RUN_STATUSES == set(sets["open_run"])
    assert store.ACTIVE_SIGNUP_STATUSES == set(sets["active_signup"])
    assert store.TERMINAL_SIGNUP_STATUSES == set(sets["terminal_signup"])
    assert store.SEATED_SIGNUP_STATUSES == set(sets["seated_signup"])
    assert store.PROMOTABLE_SIGNUP_STATUSES == set(sets["promotable_signup"])
    assert store.READY_OR_ACTIVE_SIGNUP_STATUSES == set(sets["ready_or_active_signup"])


def test_active_and_terminal_signup_sets_are_disjoint():
    """A signup is either live (occupying a seat) or terminal — never both. Guards against a typo
    that would, e.g., make an 'expired' signup still count toward run_full."""
    sets = _load_json_sets()
    assert set(sets["active_signup"]).isdisjoint(set(sets["terminal_signup"]))


def test_sql_in_list_preserves_order_and_has_no_spaces():
    """_sql_in_list renders 'a','b','c' in JSON order with no spaces — the exact text other queries
    (and the JS parity tests) match as substrings. A reorder/spacing change would silently break them."""
    sets = _load_json_sets()
    for name, members in sets.items():
        expected = ",".join(f"'{m}'" for m in members)
        assert store._sql_in_list(name) == expected
        assert " " not in store._sql_in_list(name)
    # The active-signup body is the one matched verbatim across store.py and _db.js.
    assert store._sql_in_list("active_signup") == "'waiting','ready_required','ready','active'"
    assert store._sql_in_list("terminal_signup") == "'completed','rejected','expired','cancelled'"


def test_js_reads_the_same_checked_in_file():
    """api/_db.js loads the SAME arena/statuses.json (not a copied literal). Verified by source: it
    fs-reads ../arena/statuses.json relative to its own module URL and renders IN(...) via sqlInList."""
    src = DB_JS_PATH.read_text(encoding="utf-8")
    assert "../arena/statuses.json" in src, "_db.js must read the shared arena/statuses.json"
    assert "readFileSync" in src and "sqlInList" in src
    # The inlined status literals must be GONE from _db.js (replaced by ${...IN} interpolation).
    assert "'waiting','ready_required','ready','active'" not in src, (
        "_db.js still inlines the active-signup literal instead of referencing statuses.json"
    )
