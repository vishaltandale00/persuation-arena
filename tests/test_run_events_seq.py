"""run_events.seq must be a per-run monotonic DISTINCT cursor — a duplicate makes a polling agent
skip its turn prompt and forfeit. Guards the UNIQUE(run_id,seq) index + append_event_tx's
retry-on-conflict (the HTTP thread via append_event and the coordinator thread via append_event_tx
race to allocate COALESCE(MAX(seq),0)+1)."""
import sqlite3
import threading

import pytest

from arena import store


def _sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "ev.db")
    store.init_schema()


def test_run_seq_unique_index_exists(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    with store.conn() as c:
        idx = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='run_events_run_seq_uq'"
        ).fetchone()
    assert idx is not None


def test_append_event_allocates_monotonic_distinct_seq(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    seqs = [store.append_event("r1", "e", {"i": i})["seq"] for i in range(6)]
    assert seqs == [1, 2, 3, 4, 5, 6]
    assert store.append_event("r2", "e", {})["seq"] == 1  # per-run sequence


def test_duplicate_seq_is_rejected(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    store.append_event("r1", "e", {})  # seq 1
    with pytest.raises(sqlite3.IntegrityError):
        with store.conn() as c:
            c.execute(
                "INSERT INTO run_events (id,run_id,game_instance_id,seq,visibility,target_signup_id,"
                "phase,type,payload_json,created_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("evt_dup", "r1", None, 1, "public", None, None, "e", "{}", store._utcnow()),
            )


def test_concurrent_unlocked_appends_stay_distinct(tmp_path, monkeypatch):
    """The coordinator thread calls append_event_tx directly (no _event_seq_lock). Hammer that path
    from several threads: the UNIQUE index + retry must yield distinct seqs with NO unhandled error."""
    _sqlite(tmp_path, monkeypatch)
    seqs: list[int] = []
    errors: list[Exception] = []
    guard = threading.Lock()

    def unlocked_append():
        try:
            for _ in range(8):
                with store.conn() as c:
                    s = store.append_event_tx(c, "r1", "e", {})["seq"]
                with guard:
                    seqs.append(s)
        except Exception as e:  # a non-recovered collision would land here
            with guard:
                errors.append(e)

    threads = [threading.Thread(target=unlocked_append) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"append raised under contention (retry failed): {errors}"
    assert len(seqs) == 32
    assert len(set(seqs)) == 32, "duplicate seq under concurrency"
    assert sorted(seqs) == list(range(1, 33))
