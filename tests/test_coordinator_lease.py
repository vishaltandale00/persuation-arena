"""Atomic coordinator-spawn lease (store.claim_coordinator_spawn): exactly one spawn trigger per open
connected run wins, so two coordinators never double-drive a run. Replaces the spawn endpoint's
non-atomic get_run() check-then-spawn (a TOCTOU two concurrent Vercel fetches could both pass)."""
import threading

from arena import store


def _sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "lease.db")
    store.init_schema()


def _open_run(run_id="r1", status="open"):
    store.save_run({
        "id": run_id, "game": "onuw", "label": "x", "status": status,
        "n_games": 1, "players": 5, "seed_base": 1, "created": "2026-01-01T00:00",
        "agents": [], "created_utc": "2026-01-01T00:00:00Z", "deck_preset": "arena",
    })


def test_single_claim_wins_retry_loses(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _open_run()
    assert store.claim_coordinator_spawn("r1") is True    # first trigger wins
    assert store.claim_coordinator_spawn("r1") is False   # retry / dup hits the live lease


def test_claim_blocked_once_coordinator_published(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _open_run()
    store.set_coordinator_url("r1", "https://coord")      # the container booted
    assert store.claim_coordinator_spawn("r1") is False   # already coordinated


def test_claim_blocked_when_run_not_open(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _open_run()
    store.update_run_status("r1", "active")
    assert store.claim_coordinator_spawn("r1") is False


def test_claim_reclaimable_after_lease_expiry(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _open_run()
    assert store.claim_coordinator_spawn("r1", lease_seconds=-5) is True   # set an already-expired lease
    assert store.claim_coordinator_spawn("r1") is True                     # crashed spawn -> re-claimable


def test_concurrent_claims_exactly_one_winner(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _open_run()
    results: list[bool] = []
    guard = threading.Lock()

    def claim():
        r = store.claim_coordinator_spawn("r1")
        with guard:
            results.append(r)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for r in results if r) == 1, f"expected exactly one winner, got {results}"
