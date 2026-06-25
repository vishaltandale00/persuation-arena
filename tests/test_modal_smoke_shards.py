"""P2 (codex) verifiers for the MANUAL Modal smoke path (tools/connected_modal_smoke.py).

These are the V-10 manual gate's two structural bugs, captured fast + LLM/Modal-free:

(i) Identity linkage across shards (REQ-5 / D9). Each shard has a DIFFERENT coordinator URL, and
    CredentialsStore keys saved profiles by server URL, so naively reusing one cred file across
    shards registers a SEPARATE agent_id per shard for the same displayed identity. The fix
    pre-registers each identity ONCE against the central API and reuses that one credential
    (agent_id) for every shard, regardless of the per-shard coordinator URL.

(ii) Shared DATABASE_URL (D1/D8). The smoke writes the parent/child shard rows through the LOCAL
    store; if DATABASE_URL differs from the Modal coordinators', the children exist only in local
    SQLite (and the coordinators replay the FULL schedule, so the parent rollup never completes).
    The sharded smoke must fail fast with a clear message when DATABASE_URL is unset/inconsistent
    instead of silently writing to local SQLite.
"""
from __future__ import annotations

import argparse
import json

import pytest

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.credentials import AgentCredentials, CredentialsStore

import tools.connected_modal_smoke as smoke


CENTRAL = "https://central.example"
COORD_A = "https://shard-a.modal.run"
COORD_B = "https://shard-b.modal.run"


def _fake_register(creds_by_name):
    """Build an injectable register(name, server) that hands out a fixed agent_id per display name,
    independent of the server it is asked to register against (mimics one Neon-backed central API)."""

    def register(name: str, server: str) -> AgentCredentials:
        agent_id, token = creds_by_name[name]
        return AgentCredentials(server=server.rstrip("/"), agent_id=agent_id,
                                display_name=name, agent_token=token)

    return register


def test_shared_identity_keeps_one_agent_id_across_coordinators(tmp_path):
    """P2(i) / REQ-5: ONE identity pre-registered once must present the SAME agent_id when its agent
    connects to two DIFFERENT per-shard coordinator URLs (one rating competitor across shards)."""
    cred_path = tmp_path / "id_0.json"
    register = _fake_register({"rand-0": ("agent_fixed_0", "tok_0")})

    # Pre-register the identity ONCE against the central API -> one credential (one agent_id).
    base = smoke._shared_identity_creds(
        ["rand-0"], central_server=CENTRAL, cred_paths=[str(cred_path)], register=register)
    assert base[0].agent_id == "agent_fixed_0"

    # Seed the SAME credential for each shard's coordinator URL, then resolve via the SDK's
    # ensure_registered (which keys profiles by server URL). Both coordinators must yield the SAME id.
    smoke._seed_cred_for_coordinator(base[0], COORD_A, str(cred_path))
    smoke._seed_cred_for_coordinator(base[0], COORD_B, str(cred_path))

    agent_a = ArenaAgent(name="rand-0", server=COORD_A, credentials=CredentialsStore(str(cred_path)))
    agent_b = ArenaAgent(name="rand-0", server=COORD_B, credentials=CredentialsStore(str(cred_path)))
    id_a = agent_a.ensure_registered().agent_id
    id_b = agent_b.ensure_registered().agent_id

    assert id_a == id_b == "agent_fixed_0", (
        "shared identity must keep ONE agent_id across shards (REQ-5), "
        f"got {id_a!r} on shard A and {id_b!r} on shard B")


def test_shared_identity_does_not_register_per_coordinator(tmp_path):
    """P2(i): once an identity is seeded for a coordinator URL, ensure_registered must NOT call the
    network register again — it reuses the pre-registered agent_id (the linkage linchpin)."""
    cred_path = tmp_path / "id_0.json"
    calls = {"n": 0}

    def register(name: str, server: str) -> AgentCredentials:
        calls["n"] += 1
        return AgentCredentials(server=server.rstrip("/"), agent_id=f"agent_{calls['n']}",
                                display_name=name, agent_token="tok")

    base = smoke._shared_identity_creds(
        ["rand-0"], central_server=CENTRAL, cred_paths=[str(cred_path)], register=register)
    assert calls["n"] == 1  # registered exactly ONCE against the central API
    smoke._seed_cred_for_coordinator(base[0], COORD_A, str(cred_path))
    smoke._seed_cred_for_coordinator(base[0], COORD_B, str(cred_path))

    # Two different coordinators, but no further registration: same agent_id, register() never re-hit.
    ArenaAgent(name="rand-0", server=COORD_A,
               credentials=CredentialsStore(str(cred_path))).ensure_registered()
    ArenaAgent(name="rand-0", server=COORD_B,
               credentials=CredentialsStore(str(cred_path))).ensure_registered()
    assert calls["n"] == 1, "must not re-register per shard coordinator URL"

    # The cred file holds the SAME agent_id under each coordinator's profile key.
    profiles = json.loads(cred_path.read_text())["profiles"]
    ids = {p["agent_id"] for p in profiles.values()}
    assert ids == {"agent_1"}, profiles


def _shard_args(database_url_present: bool) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="modal_smoke_shards_test", games=4, rounds=2, seed=31337,
        url_timeout=1.0, shards=2,
        central_server=CENTRAL,
    )


def test_sharded_smoke_errors_clearly_without_database_url(tmp_path, monkeypatch):
    """P2(ii) / D1: the sharded smoke must fail fast with a clear message when DATABASE_URL is unset,
    instead of silently writing the parent/child rows to a LOCAL SQLite the Modal coordinators never
    read. We must error BEFORE any run rows are written and BEFORE Modal is spawned."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Guard rails: if the smoke ever reaches create_sharded_run / Modal spawn without DATABASE_URL,
    # blow up loudly so the test can't pass for the wrong reason.
    def _boom_create(*a, **k):
        raise AssertionError("create_sharded_run reached without DATABASE_URL (silent local write)")

    monkeypatch.setattr(smoke, "create_sharded_run", _boom_create)
    monkeypatch.setattr(smoke.modal.Function, "from_name",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("Modal spawned without DATABASE_URL"))))

    with pytest.raises(SystemExit) as ei:
        smoke._sharded_smoke(_shard_args(database_url_present=False))
    msg = str(ei.value)
    assert "DATABASE_URL" in msg, msg


def test_sharded_smoke_proceeds_with_database_url(tmp_path, monkeypatch):
    """P2(ii): with DATABASE_URL set, the guard passes and the smoke gets as far as writing rows
    (we stub Modal so the test stays fast + Modal-free)."""
    monkeypatch.setenv("DATABASE_URL", "postgres://example/shared")

    created = {}

    def _spy_create(parent_cfg, k):
        created["parent"] = parent_cfg["id"]
        created["k"] = k
        return [f"{parent_cfg['id']}_shard_{i}" for i in range(k)]

    monkeypatch.setattr(smoke, "create_sharded_run", _spy_create)
    # Network-free identity pre-registration (the real SDK register would hit the central API).
    monkeypatch.setattr(smoke, "_default_register",
                        lambda name, server: AgentCredentials(
                            server=server.rstrip("/"), agent_id=f"agent_{name}",
                            display_name=name, agent_token="tok"))
    # Stop right after the DATABASE_URL guard + row creation: a sentinel from the next real step.
    monkeypatch.setattr(smoke.modal.Function, "from_name",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(_StopHere())))

    with pytest.raises(_StopHere):
        smoke._sharded_smoke(_shard_args(database_url_present=True))
    assert created["k"] == 2 and created["parent"] == "modal_smoke_shards_test"


class _StopHere(Exception):
    pass
