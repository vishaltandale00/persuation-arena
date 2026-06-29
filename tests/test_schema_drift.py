"""Schema drift guard (the CHECK, not GENERATE approach — see the schema-single-source research memo).

The column definitions are hand-typed across three surfaces: the SQLite DDL (SQLITE_SCHEMA +
_MIGRATIONS), the Postgres DDL (PG_SCHEMA_STMTS + PG_MIGRATION_STMTS), and the INSERT column lists
in api/*.js and arena/store.py. Nothing generates one from another, so they can silently diverge —
e.g. a column added to PG but not SQLite, or a JS INSERT naming a column that no longer exists.

This test introspects all three and fails loudly in CI when they disagree:
  1. SQLite columns == Postgres columns, per table (dual-backend parity).
  2. Every column named in any INSERT (JS + Python) is a real column of that table (no phantom /
     typo'd / renamed-away / one-backend-only columns).

It runs in plain pytest (no live Postgres): the SQLite side executes the real init_schema; the PG
side is parsed from the DDL text. Scope is table/column presence only; type/default/index/constraint
parity still depends on reviewing the handwritten DDL.
"""
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "api"
STORE_PY = ROOT / "arena" / "store.py"

# table-level constraint keywords that begin a non-column item inside a CREATE TABLE body
_CONSTRAINT_KW = ("primary", "unique", "foreign", "check", "constraint")


def _split_top_level_commas(body: str) -> list[str]:
    """Split on commas that are NOT inside parentheses (so 'PRIMARY KEY (a, b)' stays one item)."""
    items, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        items.append("".join(cur))
    return items


def _columns_from_create(stmt: str) -> tuple[str, set[str]] | None:
    """Parse 'CREATE TABLE [IF NOT EXISTS] <name> ( <body> )' -> (table, {column names})."""
    m = re.search(r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+([a-z_]+)\s*\((.*)\)\s*;?\s*$",
                  stmt.strip(), re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    table, body = m.group(1), m.group(2)
    cols = set()
    for item in _split_top_level_commas(body):
        tok = item.strip().split()
        if not tok:
            continue
        if tok[0].lower() in _CONSTRAINT_KW:
            continue  # table-level constraint, not a column
        cols.add(tok[0])
    return table, cols


def _pg_columns_from_ddl() -> dict[str, set[str]]:
    """Authoritative Postgres column sets, parsed from the DDL text (CREATE TABLE + ALTER ADD COLUMN)."""
    from arena import store
    tables: dict[str, set[str]] = {}
    for stmt in store.PG_SCHEMA_STMTS:
        parsed = _columns_from_create(stmt)
        if parsed:
            tables[parsed[0]] = parsed[1]
    for stmt in store.PG_MIGRATION_STMTS:
        m = re.match(r"\s*ALTER TABLE\s+([a-z_]+)\s+ADD COLUMN(?:\s+IF NOT EXISTS)?\s+([a-z_]+)",
                     stmt, re.IGNORECASE)
        if m:
            tables.setdefault(m.group(1), set()).add(m.group(2))
    return tables


def _sqlite_columns() -> dict[str, set[str]]:
    """Authoritative SQLite column sets — run the REAL init_schema (SQLITE_SCHEMA + _MIGRATIONS)."""
    # Import FIRST: arena.config runs load_dotenv() at import, which re-injects a developer .env's
    # DATABASE_URL. Pop it AFTER the import so our drop is the last word, then assert SQLite — never
    # let this drift check touch prod Neon (the same footgun gen_rating_golden.py guards against).
    from arena import store
    os.environ.pop("DATABASE_URL", None)
    assert not store._is_pg(), "schema drift check must run on SQLite, not the remote DB"
    tmp_dir = Path(tempfile.mkdtemp(prefix="schema_drift_"))
    tmp = tmp_dir / "drift.db"
    orig = store.DB_PATH
    store.DB_PATH = tmp
    try:
        store.init_schema()
        with sqlite3.connect(tmp) as c:
            c.row_factory = sqlite3.Row
            names = [r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            return {t: {r["name"] for r in c.execute(f"PRAGMA table_info({t})")} for t in names}
    finally:
        store.DB_PATH = orig
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _insert_column_lists() -> list[tuple[str, str, list[str]]]:
    """Every 'INSERT INTO <table> (<cols>)' in non-test api/*.js and arena/store.py.
    Returns (source_file, table, [columns])."""
    out: list[tuple[str, str, list[str]]] = []
    files = [p for p in API_DIR.rglob("*.js") if not p.name.endswith(".test.js")]
    files.append(STORE_PY)
    pat = re.compile(r"INSERT\s+INTO\s+([a-z_]+)\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)
    for path in files:
        text = path.read_text()
        for m in pat.finditer(text):
            table = m.group(1)
            cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
            # skip f-string interpolations / placeholders that aren't bare identifiers
            cols = [c for c in cols if re.fullmatch(r"[a-z_][a-z0-9_]*", c)]
            if cols:
                out.append((str(path.relative_to(ROOT)), table, cols))
    return out


def test_sqlite_and_postgres_columns_match():
    """Dual-backend parity: every table's column SET is identical in SQLite and Postgres DDL."""
    sqlite_cols = _sqlite_columns()
    pg_cols = _pg_columns_from_ddl()
    assert set(sqlite_cols) == set(pg_cols), (
        f"table set differs — SQLite-only: {set(sqlite_cols) - set(pg_cols)}; "
        f"PG-only: {set(pg_cols) - set(sqlite_cols)}"
    )
    mismatches = {
        t: {"sqlite_only": sorted(sqlite_cols[t] - pg_cols[t]),
            "pg_only": sorted(pg_cols[t] - sqlite_cols[t])}
        for t in sqlite_cols
        if sqlite_cols[t] != pg_cols[t]
    }
    assert not mismatches, f"SQLite/Postgres column drift: {mismatches}"


def test_every_insert_targets_a_real_column():
    """No INSERT (JS or Python) may name a column absent from the schema (catches typos, renames,
    one-backend-only columns). Uses the SQLite canonical set (== PG by the parity test)."""
    schema = _sqlite_columns()
    inserts = _insert_column_lists()
    assert inserts, "no INSERT statements found — the scanner regressed"
    errors = []
    for source, table, cols in inserts:
        known = schema.get(table)
        if known is None:
            errors.append(f"{source}: INSERT INTO unknown table '{table}'")
            continue
        phantom = [c for c in cols if c not in known]
        if phantom:
            errors.append(f"{source}: INSERT INTO {table} names non-existent column(s) {phantom}")
    assert not errors, "INSERT/schema drift:\n  " + "\n  ".join(errors)


def test_scanner_sees_known_inserts():
    """Guard the guard: the scanner must actually be finding the JS + Python INSERTs (so a parser
    regression can't make the drift checks vacuously pass)."""
    inserts = _insert_column_lists()
    tables_seen = {t for _, t, _ in inserts}
    js_sources = {s for s, _, _ in inserts if s.startswith("api/")}
    assert {"runs", "ratings", "game_players"} <= tables_seen, f"missing expected tables: {tables_seen}"
    assert js_sources, "no api/*.js INSERTs scanned"
