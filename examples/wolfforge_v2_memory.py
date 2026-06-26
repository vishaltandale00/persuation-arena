"""Optional, safety-scoped cross-run memory for WolfForgeAgentV2.

This is a FOLLOW-UP capability, OFF BY DEFAULT, and not part of the original V2 performance claim.
WolfForgeV2's default guarantee is unchanged: stable identity + isolated per-game state, NOT cross-run
learning. Turning this on (WOLFFORGE_V2_MEMORY_MODE != off) lets the agent accumulate *safe, compact,
post-game summaries* across completed games and surface them to the model as a clearly-separated WEAK
PRIOR — never as current-game facts.

Hard safety boundary — long-term memory stores ONLY:
  - the agent's own dealt/believed role and own win/loss;
  - the agent's own reliability counters (fallbacks/repairs/timeouts);
  - finite-taxonomy failure labels;
  - machine-generated lessons from FIXED templates keyed by (role, objective, label);
  - aggregate "games seen" counts for stable public opponents.
It NEVER stores: raw private night observations, full transcripts, raw model output, full prompts,
API keys / tokens / credentials, hidden role facts about other players, or any free player text
(so an opponent's in-game prompt injection can never become a long-term instruction).

Everything here is model-free and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
from typing import Any

SCHEMA_VERSION = 1

# --- modes ----------------------------------------------------------------------------------------
_MODES = ("off", "read", "write", "readwrite")


def normalize_mode(mode: str | None) -> str:
    m = (mode or "off").strip().lower()
    return m if m in _MODES else "off"


def _can_read(mode: str) -> bool:
    return mode in ("read", "readwrite")


def _can_write(mode: str) -> bool:
    return mode in ("write", "readwrite")


# --- role / objective taxonomy --------------------------------------------------------------------
_EVIL_ROLES = {"Werewolf", "Minion"}
FAILURE_LABELS = ("none", "timeout", "reliability", "tanner_strategy", "minion_strategy",
                  "late_vote", "mechanical", "weak_coalition", "play")


def objective_group(role: str | None) -> str:
    """Coarse objective bucket for a believed role (deterministic)."""
    if not role:
        return "unknown"
    if role == "Werewolf":
        return "werewolf"
    if role == "Minion":
        return "minion"
    if role == "Tanner":
        return "tanner"
    return "village"


def _team_of(role: str | None) -> str:
    if role in _EVIL_ROLES:
        return "evil"
    if role == "Tanner":
        return "tanner"
    return "good"


def did_i_win(role: str | None, seat: int | None, winner_team: str | None,
              deaths: list | None) -> bool:
    """Deterministic self-assessed win from a game_result payload + the agent's believed role/seat.

    Heuristic by design (based on the agent's *believed* role, which Robber/Drunk swaps can make
    approximate); used only to bucket the agent's own learning, never as game-truth."""
    deaths = deaths or []
    if role == "Tanner":
        return seat in deaths
    if winner_team in (None, "void"):
        return False
    return _team_of(role) == winner_team


def failure_label(*, won: bool, role: str | None, fallback_count: int, repair_count: int,
                  timeouts: int) -> str:
    """Pick one finite-taxonomy failure label from deterministic counters (no transcript mining)."""
    if won:
        return "none"
    if timeouts > 0:
        return "timeout"
    if fallback_count > 0 or repair_count > 0:
        return "reliability"
    if role == "Tanner":
        return "tanner_strategy"
    if role == "Minion":
        return "minion_strategy"
    return "play"


# --- value types ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpponentIdentity:
    """A public opponent identity. agent_id is the only STABLE key; without it we fall back to a
    namespaced display-name hash and mark confidence low."""
    display_name: str
    agent_id: str | None = None

    def key(self) -> str:
        if self.agent_id:
            return f"agent:{self.agent_id}"
        h = hashlib.sha256(("name:" + (self.display_name or "")).encode()).hexdigest()[:16]
        return f"name:{h}"

    def stable(self) -> bool:
        return bool(self.agent_id)


@dataclass
class CompletedGameSummary:
    run_id: str | None
    game_id: str | None
    my_role: str | None
    won: bool
    fallback_count: int = 0
    repair_count: int = 0
    timeouts: int = 0
    tokens_used: int = 0
    opponents: list[OpponentIdentity] = field(default_factory=list)
    # Safe, aggregate-only summaries. summary_private_safe must NOT contain other players' hidden
    # facts; it is phrased as the agent's own learning. Neither is ever replayed verbatim in a prompt.
    summary_public: str = ""
    summary_private_safe: str = ""

    @property
    def my_objective_group(self) -> str:
        return objective_group(self.my_role)

    @property
    def label(self) -> str:
        return failure_label(won=self.won, role=self.my_role, fallback_count=self.fallback_count,
                             repair_count=self.repair_count, timeouts=self.timeouts)


@dataclass
class MemoryContext:
    """The bounded, structured weak-prior block injected into the prompt under `long_term_memory`."""
    role_lessons: list[str] = field(default_factory=list)
    opponent_hints: list[str] = field(default_factory=list)
    reliability_reminders: list[str] = field(default_factory=list)
    do_not_overweight: str = ("These are weak historical priors, not current-game facts.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_lessons": list(self.role_lessons),
            "opponent_hints": list(self.opponent_hints),
            "reliability_reminders": list(self.reliability_reminders),
            "do_not_overweight": self.do_not_overweight,
        }

    def items(self) -> int:
        return len(self.role_lessons) + len(self.opponent_hints) + len(self.reliability_reminders)

    def is_empty(self) -> bool:
        return self.items() == 0


# Fixed lesson templates keyed by failure label — NEVER derived from free player text, so a prompt
# injection can never be promoted into a lesson.
_LESSON_TEMPLATES = {
    "tanner_strategy": "As Tanner, avoid being too obvious; seek plausible suspicion rather than a blatant self-tell.",
    "minion_strategy": "As Minion, protect the wolves and bait a village mis-vote; never reveal which players are wolves.",
    "reliability": "Reliability: emit a valid, schema-correct action promptly; avoid risky long outputs near the deadline.",
    "late_vote": "Coordinate the vote earlier; name one explicit target before the buzzer.",
    "weak_coalition": "Build a concrete coalition around a shared, repeatable plan instead of vague suspicion.",
    "mechanical": "Re-check night-action chronology and never make a mechanically impossible claim.",
    "play": "Anchor on the strongest verifiable claim and converge the table on one target.",
}


def default_memory_path() -> Path:
    """Cross-platform per-user data path (no repo-specific helper exists for this)."""
    override = os.environ.get("WOLFFORGE_V2_MEMORY_PATH")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "persuasion-arena" / "wolfforge_v2_memory.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_meta (
  k TEXT PRIMARY KEY, v TEXT
);
CREATE TABLE IF NOT EXISTS role_objective_stats (
  role TEXT, objective_group TEXT, games INTEGER DEFAULT 0, wins INTEGER DEFAULT 0,
  losses INTEGER DEFAULT 0, fallbacks INTEGER DEFAULT 0, repairs INTEGER DEFAULT 0,
  timeouts INTEGER DEFAULT 0, late_vote_failures INTEGER DEFAULT 0, tanner_failures INTEGER DEFAULT 0,
  minion_failures INTEGER DEFAULT 0, mechanical_failures INTEGER DEFAULT 0,
  weak_coalition_failures INTEGER DEFAULT 0, updated_at TEXT,
  PRIMARY KEY (role, objective_group)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS opponent_profiles (
  opponent_key TEXT PRIMARY KEY, display_name_last_seen TEXT, games_seen INTEGER DEFAULT 0,
  times_claimed_role INTEGER DEFAULT 0, claim_volatility_score REAL DEFAULT 0,
  late_target_switches INTEGER DEFAULT 0, often_leads_votes INTEGER DEFAULT 0,
  often_follows_votes INTEGER DEFAULT 0, observed_bluff_suspicions INTEGER DEFAULT 0,
  cooperation_score REAL DEFAULT 0, deception_risk_score REAL DEFAULT 0,
  confidence REAL DEFAULT 0, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS policy_lessons (
  lesson_id TEXT PRIMARY KEY, scope TEXT, role TEXT, objective_group TEXT, label TEXT, text TEXT,
  support_count REAL DEFAULT 0, contradiction_count REAL DEFAULT 0, confidence REAL DEFAULT 0,
  created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS game_summaries (
  run_id TEXT, game_id TEXT, my_role TEXT, my_objective_group TEXT, won INTEGER,
  summary_public TEXT, summary_private_safe TEXT, failure_label TEXT, tokens_used INTEGER,
  fallback_count INTEGER, repair_count INTEGER, created_at TEXT
);
"""


class CrossRunMemory:
    """SQLite-backed cross-run memory. Safe to share across threads (one lock-guarded connection).

    On any corruption / IO error it DISABLES itself (mode -> off) and warns, never crashing a game.
    """

    def __init__(self, path: str | Path | None = None, *, mode: str = "off",
                 max_prompt_chars: int = 1200, min_games_for_opponent_hint: int = 3,
                 decay: float = 0.90, policy_version: str | None = None,
                 prompt_hash: str | None = None, now: str = "1970-01-01T00:00:00Z"):
        self.mode = normalize_mode(mode)
        self.path = Path(path).expanduser() if path else default_memory_path()
        self.max_prompt_chars = int(max_prompt_chars)
        self.min_games_for_opponent_hint = int(min_games_for_opponent_hint)
        self.decay = float(decay)
        self.policy_version = policy_version
        self.prompt_hash = prompt_hash
        self._now = now
        self.disabled = False
        self.write_count = 0
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.mode != "off":
            self._open()

    # ---- lifecycle ----------------------------------------------------------------------------
    def _open(self) -> None:
        try:
            if self.path != Path(":memory:"):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO memory_meta(k, v) VALUES('schema_version', ?)",
                               (str(SCHEMA_VERSION),))
            self._conn.execute("INSERT OR IGNORE INTO memory_meta(k, v) VALUES('created_at', ?)",
                               (self._now,))
            for k, v in (("agent_policy_version", self.policy_version or ""),
                         ("prompt_hash", self.prompt_hash or ""), ("updated_at", self._now)):
                self._conn.execute("INSERT OR REPLACE INTO memory_meta(k, v) VALUES(?, ?)", (k, v))
            self._conn.commit()
        except (sqlite3.Error, OSError) as e:  # corrupt DB / unwritable path -> disable, do not crash
            self._disable(f"{type(e).__name__}: {e}")

    def _disable(self, reason: str) -> None:
        self.disabled = True
        self.mode = "off"
        try:
            if self._conn:
                self._conn.close()
        except sqlite3.Error:
            pass
        self._conn = None
        print(f"[wolfforge-v2-memory] disabled (memory off): {reason}", flush=True)

    def close(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def can_read(self) -> bool:
        return _can_read(self.mode) and not self.disabled and self._conn is not None

    def can_write(self) -> bool:
        return _can_write(self.mode) and not self.disabled and self._conn is not None

    # ---- read ---------------------------------------------------------------------------------
    def load_context(self, *, my_role: str | None, objective_group: str | None,
                     visible_opponents: list[OpponentIdentity] | None = None) -> MemoryContext:
        ctx = MemoryContext()
        if not self.can_read():
            return ctx
        og = objective_group or globals()["objective_group"](my_role)
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT text, confidence FROM policy_lessons "
                    "WHERE (role=? OR role='' OR role IS NULL) AND (objective_group=? OR objective_group='*') "
                    "ORDER BY confidence DESC, lesson_id LIMIT 5",
                    (my_role or "", og)).fetchall()
                ctx.role_lessons = [r["text"] for r in rows if r["text"]]
                rel = self._conn.execute(
                    "SELECT games, fallbacks, repairs FROM role_objective_stats WHERE role=?",
                    (my_role or "",)).fetchone()
                if rel and rel["games"] and (rel["fallbacks"] + rel["repairs"]) > 0:
                    rate = (rel["fallbacks"] + rel["repairs"]) / rel["games"]
                    if rate >= 0.15:
                        ctx.reliability_reminders.append(
                            "Historically you have a higher fallback/repair rate in this role — keep "
                            "actions short and schema-valid.")
                for opp in (visible_opponents or []):
                    prof = self._conn.execute(
                        "SELECT games_seen, confidence FROM opponent_profiles WHERE opponent_key=?",
                        (opp.key(),)).fetchone()
                    if prof and prof["games_seen"] >= self.min_games_for_opponent_hint:
                        conf = "stable-id" if opp.stable() else "low-confidence name match"
                        ctx.opponent_hints.append(
                            f"You have shared {prof['games_seen']} prior games with {opp.display_name} "
                            f"({conf}). Weak prior only; rely on this game's behavior.")
        except sqlite3.Error as e:
            self._disable(f"{type(e).__name__}: {e}")
            return MemoryContext()
        return _cap_context(ctx, self.max_prompt_chars)

    # ---- write --------------------------------------------------------------------------------
    def record_completed_game(self, *, game_summary: CompletedGameSummary) -> None:
        if not self.can_write():
            return
        g = game_summary
        role, og, label = g.my_role or "", g.my_objective_group, g.label
        try:
            with self._lock:
                c = self._conn
                c.execute(
                    "INSERT INTO game_summaries(run_id,game_id,my_role,my_objective_group,won,"
                    "summary_public,summary_private_safe,failure_label,tokens_used,fallback_count,"
                    "repair_count,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (g.run_id, g.game_id, role, og, 1 if g.won else 0,
                     _safe_text(g.summary_public), _safe_text(g.summary_private_safe), label,
                     int(g.tokens_used), int(g.fallback_count), int(g.repair_count), self._now))
                c.execute("INSERT OR IGNORE INTO role_objective_stats(role,objective_group,updated_at) "
                          "VALUES(?,?,?)", (role, og, self._now))
                col = {"tanner_strategy": "tanner_failures", "minion_strategy": "minion_failures",
                       "late_vote": "late_vote_failures", "mechanical": "mechanical_failures",
                       "weak_coalition": "weak_coalition_failures"}.get(label)
                extra = f", {col}={col}+1" if col else ""
                c.execute(
                    f"UPDATE role_objective_stats SET games=games+1, wins=wins+?, losses=losses+?, "
                    f"fallbacks=fallbacks+?, repairs=repairs+?, timeouts=timeouts+?{extra}, updated_at=? "
                    f"WHERE role=? AND objective_group=?",
                    (1 if g.won else 0, 0 if g.won else 1, int(g.fallback_count), int(g.repair_count),
                     int(g.timeouts), self._now, role, og))
                for opp in g.opponents:
                    c.execute("INSERT OR IGNORE INTO opponent_profiles(opponent_key,updated_at) "
                              "VALUES(?,?)", (opp.key(), self._now))
                    conf = 0.6 if opp.stable() else 0.2
                    c.execute("UPDATE opponent_profiles SET games_seen=games_seen+1, "
                              "display_name_last_seen=?, confidence=?, updated_at=? WHERE opponent_key=?",
                              (opp.display_name, conf, self._now, opp.key()))
                self._refresh_lessons(role, og)
                c.execute("INSERT OR REPLACE INTO memory_meta(k,v) VALUES('updated_at', ?)", (self._now,))
                c.commit()
            self.write_count += 1
        except sqlite3.Error as e:
            self._disable(f"{type(e).__name__}: {e}")

    def _refresh_lessons(self, role: str, og: str) -> None:
        """Regenerate lessons for (role, og) from deterministic counters + FIXED templates."""
        c = self._conn
        row = c.execute("SELECT * FROM role_objective_stats WHERE role=? AND objective_group=?",
                        (role, og)).fetchone()
        if not row:
            return
        games = row["games"] or 0
        candidates: list[tuple[str, float]] = []  # (label, support)
        if row["tanner_failures"]:
            candidates.append(("tanner_strategy", row["tanner_failures"]))
        if row["minion_failures"]:
            candidates.append(("minion_strategy", row["minion_failures"]))
        if (row["fallbacks"] + row["repairs"]) > 0:
            candidates.append(("reliability", row["fallbacks"] + row["repairs"]))
        if row["late_vote_failures"]:
            candidates.append(("late_vote", row["late_vote_failures"]))
        if row["weak_coalition_failures"]:
            candidates.append(("weak_coalition", row["weak_coalition_failures"]))
        if row["mechanical_failures"]:
            candidates.append(("mechanical", row["mechanical_failures"]))
        for label, support in candidates:
            text = _LESSON_TEMPLATES.get(label)
            if not text:
                continue
            lid = f"{role}|{og}|{label}"
            prev = c.execute("SELECT support_count FROM policy_lessons WHERE lesson_id=?", (lid,)).fetchone()
            # Decayed accumulation: old support fades by `decay`, current observation set as support.
            support_count = (prev["support_count"] * self.decay if prev else 0.0) + float(support)
            confidence = round(min(1.0, support_count / max(1.0, float(games))), 4)
            c.execute(
                "INSERT INTO policy_lessons(lesson_id,scope,role,objective_group,label,text,"
                "support_count,contradiction_count,confidence,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,0,?,?,?) ON CONFLICT(lesson_id) DO UPDATE SET "
                "support_count=excluded.support_count, confidence=excluded.confidence, "
                "text=excluded.text, updated_at=excluded.updated_at",
                (lid, "role", role, og, label, text, round(support_count, 4), confidence,
                 self._now, self._now))

    # ---- export / hash / prune ----------------------------------------------------------------
    def export_compact(self) -> dict[str, Any]:
        """A deterministic, timestamp-free snapshot suitable for hashing and inspection."""
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
                               "role_objective_stats": [], "policy_lessons": [], "opponent_profiles": []}
        if self._conn is None:
            return out
        try:
            with self._lock:
                for r in self._conn.execute(
                        "SELECT role,objective_group,games,wins,losses,fallbacks,repairs,timeouts,"
                        "late_vote_failures,tanner_failures,minion_failures,mechanical_failures,"
                        "weak_coalition_failures FROM role_objective_stats ORDER BY role,objective_group"):
                    out["role_objective_stats"].append(dict(r))
                for r in self._conn.execute(
                        "SELECT lesson_id,scope,role,objective_group,label,text,"
                        "round(support_count,4) support_count, round(confidence,4) confidence "
                        "FROM policy_lessons ORDER BY lesson_id"):
                    out["policy_lessons"].append(dict(r))
                for r in self._conn.execute(
                        "SELECT opponent_key,games_seen,round(confidence,4) confidence "
                        "FROM opponent_profiles ORDER BY opponent_key"):
                    out["opponent_profiles"].append(dict(r))
        except sqlite3.Error as e:
            self._disable(f"{type(e).__name__}: {e}")
        return out

    def snapshot_hash(self) -> str:
        """sha256 over schema_version + a sorted, timestamp-free compact export. Stable across
        process restarts for identical content."""
        payload = json.dumps(self.export_compact(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def prune(self, *, min_confidence: float = 0.0, decay: float | None = None) -> int:
        """Deterministically decay lesson support and drop lessons below a confidence floor.
        Returns the number of lessons removed."""
        if not self.can_write():
            return 0
        d = self.decay if decay is None else float(decay)
        removed = 0
        try:
            with self._lock:
                c = self._conn
                for r in c.execute("SELECT lesson_id, support_count, role, objective_group "
                                   "FROM policy_lessons").fetchall():
                    games_row = c.execute("SELECT games FROM role_objective_stats WHERE role=? AND "
                                          "objective_group=?", (r["role"], r["objective_group"])).fetchone()
                    games = (games_row["games"] if games_row else 0) or 1
                    new_support = r["support_count"] * d
                    conf = min(1.0, new_support / games)
                    if conf < min_confidence:
                        c.execute("DELETE FROM policy_lessons WHERE lesson_id=?", (r["lesson_id"],))
                        removed += 1
                    else:
                        c.execute("UPDATE policy_lessons SET support_count=?, confidence=? WHERE lesson_id=?",
                                  (round(new_support, 4), round(conf, 4), r["lesson_id"]))
                c.commit()
        except sqlite3.Error as e:
            self._disable(f"{type(e).__name__}: {e}")
        return removed


def _safe_text(text: str, limit: int = 400) -> str:
    """Belt-and-braces: collapse whitespace and cap length. Callers must already pass safe summaries
    (no hidden facts / no raw player text); this only bounds size."""
    return " ".join(str(text or "").split())[:limit]


def _cap_context(ctx: MemoryContext, max_chars: int) -> MemoryContext:
    """Bound the injected memory to max_chars by trimming items (lessons first kept, hints last)."""
    while ctx.items() > 0 and len(json.dumps(ctx.to_dict(), sort_keys=True)) > max_chars:
        if ctx.opponent_hints:
            ctx.opponent_hints.pop()
        elif ctx.reliability_reminders:
            ctx.reliability_reminders.pop()
        elif ctx.role_lessons:
            ctx.role_lessons.pop()
        else:
            break
    return ctx
