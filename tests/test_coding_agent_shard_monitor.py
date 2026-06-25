"""P2d verifier for the --shards block-monitor (tools/coding_agent_run.py:_monitor_shards).

BUG (P2d): if a shard-host subprocess exits before its child run reaches a terminal
('done'/'partial') state — e.g. `_run_shard_child` returns nonzero because not enough agents
activated, leaving the child 'waiting'/'ready_required' — then all subprocesses are dead but
rollup_parent_status stays 'running'. The old monitor's loop only broke when the rollup was already
terminal AND all procs had exited, so with a dead host + non-terminal child it would spin until the
full games*rounds*deadline timeout (minutes/hours) before giving up.

FIX: the monitor must notice when every shard-host subprocess has exited and stop waiting promptly —
resolving the unfinished child(ren) to 'partial' (so the rollup becomes terminal) and returning —
instead of looping to the long timeout. The happy path (all children reach 'done') is preserved.

These tests are LLM/subprocess-free: a fake clock makes "promptly" assertable without real sleeping,
and a FakeProc stands in for subprocess.Popen.
"""
from __future__ import annotations

import types

import tools.coding_agent_run as car


class FakeProc:
    """Stand-in for subprocess.Popen. poll() returns None until `exit(rc)` is called, then `rc`."""

    def __init__(self):
        self._rc: int | None = None
        self.terminated = False

    def exit(self, rc: int = 0) -> None:
        self._rc = rc

    def poll(self):
        return self._rc

    def terminate(self) -> None:
        self.terminated = True


class FakeClock:
    """Monkeypatchable replacement for the module's `time`: time() reads a virtual clock that only
    advances when sleep() is called (so no wall-clock time passes and we can assert elapsed virtual
    time precisely). `on_tick(t)` lets a test mutate the world as virtual time progresses."""

    def __init__(self, on_tick=None):
        self.now = 0.0
        self.sleeps = 0
        self._on_tick = on_tick

    def time(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.sleeps += 1
        self.now += dt
        if self._on_tick is not None:
            self._on_tick(self.now)


def _patch_clock(monkeypatch, clock: FakeClock):
    monkeypatch.setattr(car, "time", types.SimpleNamespace(time=clock.time, sleep=clock.sleep))


# A long timeout that mimics the real games*rounds*deadline budget: if the monitor ever waits this
# out we want the test to FAIL by being obviously wrong (it should resolve in well under this).
LONG_TIMEOUT = 4 * 6 * 240.0  # 5760 virtual seconds


def test_monitor_returns_promptly_when_host_exits_with_child_not_done(monkeypatch):
    """P2d: a host subprocess has exited (nonzero) but its child never reached done/partial, so the
    rollup is stuck at 'running'. The monitor must detect the dead host, resolve the child to
    'partial', and return PROMPTLY — not loop out the entire LONG_TIMEOUT."""
    parent = "p"
    children = ["p_shard_0"]
    proc = FakeProc()
    proc.exit(1)  # host died nonzero immediately (not enough agents activated)

    marked: dict[str, str] = {}
    child_status = {"p_shard_0": "running"}

    def fake_get_run(run_id: str):
        return {"id": run_id, "status": child_status.get(run_id, "open")}

    def fake_update_status(run_id: str, status: str) -> None:
        marked[run_id] = status
        child_status[run_id] = status

    # Child status is driven by what the monitor records: stuck 'running' until resolved 'partial'.
    def fake_rollup(_parent: str) -> str:
        if marked.get("p_shard_0") == "partial":
            return "partial"
        return "running"

    monkeypatch.setattr(car.store, "get_run", fake_get_run)
    monkeypatch.setattr(car.store, "update_run_status", fake_update_status)
    monkeypatch.setattr(car, "rollup_parent_status", fake_rollup)
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)

    status = car._monitor_shards(parent, children, [proc], LONG_TIMEOUT)

    # Resolved, not hung: well under the long timeout of virtual seconds.
    assert clock.now < LONG_TIMEOUT, (
        f"monitor looped to the long timeout (virtual now={clock.now}) instead of detecting the "
        f"dead host and resolving the child")
    assert clock.now <= 30.0, f"monitor should resolve within seconds, virtual now={clock.now}"
    # The unfinished child was resolved to 'partial' so the rollup becomes terminal.
    assert marked.get("p_shard_0") == "partial", marked
    assert status == "partial", status


def test_monitor_returns_done_on_happy_path(monkeypatch):
    """Happy path preserved: all hosts exit cleanly and the children roll up to 'done' -> 'done'."""
    parent = "p"
    children = ["p_shard_0", "p_shard_1"]
    procs = [FakeProc(), FakeProc()]

    # The hosts finish (and the children reach done) a couple of ticks in.
    def on_tick(now: float) -> None:
        if now >= 2.0:
            for pr in procs:
                pr.exit(0)

    def fake_rollup(_parent: str) -> str:
        return "done" if all(pr.poll() == 0 for pr in procs) else "running"

    marked: dict[str, str] = {}
    monkeypatch.setattr(car.store, "update_run_status",
                        lambda rid, st: marked.__setitem__(rid, st))
    monkeypatch.setattr(car, "rollup_parent_status", fake_rollup)
    clock = FakeClock(on_tick=on_tick)
    _patch_clock(monkeypatch, clock)

    status = car._monitor_shards(parent, children, procs, LONG_TIMEOUT)

    assert status == "done", status
    assert clock.now < LONG_TIMEOUT
    # Happy path must NOT force any child to 'partial'.
    assert marked == {}, f"happy path must not mark any child partial, got {marked}"


def test_monitor_resolves_when_all_hosts_exit_even_if_some_child_done(monkeypatch):
    """Mixed: one host finished its child ('done'), the other died with its child stuck. All procs are
    dead, so the monitor resolves the unfinished child to 'partial' and returns 'partial' promptly."""
    parent = "p"
    children = ["p_shard_0", "p_shard_1"]
    good, bad = FakeProc(), FakeProc()
    good.exit(0)
    bad.exit(1)

    child_status = {"p_shard_0": "done", "p_shard_1": "running"}
    marked: dict[str, str] = {}

    def fake_get_run(run_id: str):
        return {"id": run_id, "status": child_status.get(run_id, "open")}

    def fake_update_status(run_id: str, status: str) -> None:
        marked[run_id] = status
        child_status[run_id] = status

    def fake_rollup(_parent: str) -> str:
        vals = list(child_status.values())
        if any(v == "partial" for v in vals):
            return "partial"
        if all(v == "done" for v in vals):
            return "done"
        return "running"

    monkeypatch.setattr(car.store, "get_run", fake_get_run)
    monkeypatch.setattr(car.store, "update_run_status", fake_update_status)
    monkeypatch.setattr(car, "rollup_parent_status", fake_rollup)
    clock = FakeClock()
    _patch_clock(monkeypatch, clock)

    status = car._monitor_shards(parent, children, [good, bad], LONG_TIMEOUT)

    assert status == "partial", status
    assert clock.now <= 30.0, f"should resolve promptly, virtual now={clock.now}"
    # Only the stuck child gets forced partial; the already-done child is left alone.
    assert marked.get("p_shard_1") == "partial", marked
    assert marked.get("p_shard_0") != "partial", marked
