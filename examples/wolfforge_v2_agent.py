"""WolfForgeAgentV2 — a persistent, connected ONUW harness.

Wire it onto the official SDK ArenaAgent (the SDK delivers events + turns and owns the event cursor;
this harness owns per-game memory and the decision policy):

    from persuasion_arena_agent.agent import ArenaAgent
    from examples.wolfforge_v2_agent import WolfForgeV2Agent

    harness = WolfForgeV2Agent(run_id="my_run")
    agent = ArenaAgent(name="WolfForgeV2", server=SERVER)
    agent.on_event(harness.on_event)   # fold each delta event into the right game's state
    agent.act(harness.act)             # decide from that state, under the turn deadline
    signup = agent.signup(run_id="my_run")
    agent.run_forever([signup])

Or run it module-style with the CLI (model/run via env + flags):

    arena-agent play --run my_run --server http://127.0.0.1:8000 \
        --name WolfForgeV2 examples/wolfforge_v2_agent.py

The reliability ladder (strict structured output -> local validation -> one repair -> deterministic
legal fallback) lives here; the model-free policy it depends on lives in wolfforge_v2_policy.py.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from arena.identity import NO_ONE_REF
from examples.wolfforge_v2_policy import (
    BASELINE_POLICY_VERSION,
    DEFAULT_MODEL,
    GameState,
    POLICY_VERSION,
    baseline_prompt_hash,
    build_messages,
    charisma_baseline_system_prompt,
    fallback_action,
    interpret,
    prompt_hash,
    response_schema,
    system_prompt,
)

# Providers whose OpenRouter route honors strict json_schema structured output. Mirrors
# arena.openrouter._STRUCTURED_SCHEMA_PROVIDERS so behavior is consistent across the repo.
_STRUCTURED_SCHEMA_PROVIDERS = {"anthropic", "google", "mistralai", "openai", "x-ai", "z-ai"}
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class V2Config:
    """Resolved agent configuration. Defaults are explicit and documented in WOLFFORGE_AGENT_V2.md.

    The temperature default (0.35) matches the original WolfForge charisma evaluation block so V2 is
    directly comparable; gpt-4o-mini keeps the model fixed to that experiment too. A stronger model
    is selectable purely through WOLFFORGE_V2_MODEL with no source change.
    """
    model: str = DEFAULT_MODEL
    temperature: float = 0.35
    reasoning_effort: str = "medium"
    max_tokens: int = 4000
    brain_timeout_s: float = 40.0
    # Time reserved for network submission + safety; below this remaining budget we never start a
    # model (or repair) call and fall back deterministically instead.
    submit_margin_s: float = 3.0
    structured_output: str = "auto"  # off | auto | json_object | json_schema
    log_path: str | None = None

    @classmethod
    def from_env(cls) -> "V2Config":
        effort = _env("WOLFFORGE_V2_REASONING_EFFORT", "medium").strip().lower()
        if effort not in _REASONING_EFFORTS:
            effort = "medium"
        so = _env("WOLFFORGE_V2_STRUCTURED_OUTPUT", "auto").strip().lower()
        if so not in {"off", "auto", "json_object", "json_schema"}:
            so = "auto"
        return cls(
            model=_env("WOLFFORGE_V2_MODEL", DEFAULT_MODEL),
            temperature=_env_float("WOLFFORGE_V2_TEMPERATURE", 0.35),
            reasoning_effort=effort,
            max_tokens=_env_int("WOLFFORGE_V2_MAX_TOKENS", 4000),
            brain_timeout_s=_env_float("WOLFFORGE_V2_BRAIN_TIMEOUT", 40.0),
            submit_margin_s=_env_float("WOLFFORGE_V2_SUBMIT_MARGIN_S", 3.0),
            structured_output=so,
            log_path=os.environ.get("WOLFFORGE_V2_LOG_PATH") or None,
        )


@dataclass
class BrainResult:
    """What a model brain returns. Fakes in tests implement the same shape (see test_*_agent.py)."""
    content: str
    ok: bool = True
    error_type: str | None = None
    resolved_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    structured_rejected: bool = False


class OpenRouterBrain:
    """Default brain: OpenRouter via the OpenAI-compatible client. Built lazily so importing this
    module (and running the model-free tests) needs no API key."""

    def __init__(self, config: V2Config):
        self.config = config
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set in the harness's environment")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        return self._client

    def complete(self, messages: list[dict], *, response_format: dict | None,
                 timeout: float) -> BrainResult:
        cfg = self.config
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "timeout": timeout,
            "extra_body": {"reasoning": {"effort": cfg.reasoning_effort, "exclude": True}},
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — classify, never crash the turn
            if response_format is not None and _looks_like_structured_rejection(e):
                # Structured output rejected by the provider: signal a clean retry without it.
                return BrainResult(content="", ok=False, error_type="structured_rejected",
                                   structured_rejected=True)
            return BrainResult(content="", ok=False, error_type=type(e).__name__)
        try:
            msg = resp.choices[0].message
            content = (getattr(msg, "content", None) or "").strip()
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            return BrainResult(content=content, ok=True,
                               resolved_model=getattr(resp, "model", None),
                               prompt_tokens=pt, completion_tokens=ct)
        except Exception as e:  # noqa: BLE001
            return BrainResult(content="", ok=False, error_type=type(e).__name__)


def _looks_like_structured_rejection(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = ("response_format", "json_schema", "json_object", "structured output",
               "structured outputs", "unsupported parameter", "unexpected keyword argument")
    return any(n in text for n in needles)


def _parse_deadline(deadline_at: str | None) -> _dt.datetime | None:
    if not deadline_at:
        return None
    try:
        return _dt.datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class WolfForgeV2Agent:
    """Persistent connected harness: isolated compact state per game, coalition-first policy, strict
    reliability ladder, deadline-aware, with safe JSONL telemetry."""

    # Declared identity metadata surfaced on registration/leaderboard.
    model = DEFAULT_MODEL
    harness = "wolfforge-v2"

    def __init__(self, run_id: str | None = None, *, config: V2Config | None = None,
                 brain: Any | None = None, agent_name: str = "WolfForgeV2",
                 now: Callable[[], _dt.datetime] | None = None,
                 system_prompt_text: str | None = None,
                 policy_version: str = POLICY_VERSION,
                 prompt_hash_value: str | None = None,
                 harness: str = "wolfforge-v2"):
        self.run_id = run_id
        self.config = config or V2Config.from_env()
        self.model = self.config.model
        self.agent_name = agent_name
        self.harness = harness
        self.system_prompt_text = system_prompt_text or system_prompt()
        self.policy_version = policy_version
        self.prompt_hash_value = prompt_hash_value or prompt_hash()
        self.brain = brain if brain is not None else OpenRouterBrain(self.config)
        self._now = now or (lambda: _dt.datetime.now(_dt.UTC))
        self.games: dict[str, GameState] = {}

    @classmethod
    def charisma_baseline(cls, run_id: str | None = None, **kwargs: Any) -> "WolfForgeV2Agent":
        """The frozen connected baseline: identical machinery, the Charisma strategy prompt. The only
        variable vs V2 is the strategy text, matching the original WolfForge ablation design."""
        kwargs.setdefault("agent_name", "CharismaBaseline")
        return cls(run_id, system_prompt_text=charisma_baseline_system_prompt(),
                   policy_version=BASELINE_POLICY_VERSION, prompt_hash_value=baseline_prompt_hash(),
                   harness="charisma-baseline", **kwargs)

    # ---- state isolation -----------------------------------------------------------------------
    def state_key(self, game_instance_id: str | None) -> str:
        """The per-game key. game_instance_id is the SDK's canonical per-game identifier (present on
        both Event and Turn); run_id scopes it across concurrent runs. game_instance_id already embeds
        the run id in practice, so it is the primary key and run_id is a defensive prefix."""
        gid = game_instance_id or "_run_level_"
        return f"{self.run_id}:{gid}" if self.run_id else gid

    def _state(self, game_instance_id: str | None) -> GameState:
        key = self.state_key(game_instance_id)
        st = self.games.get(key)
        if st is None:
            st = GameState(game_id=game_instance_id or key, run_id=self.run_id)
            self.games[key] = st
        return st

    # ---- SDK hooks -----------------------------------------------------------------------------
    def on_event(self, event: Any) -> None:
        """Fold one delivered event into the owning game's state. Run-level events (no
        game_instance_id) carry no per-game memory and are skipped. Private events the server
        delivers are always for this signup, and we key strictly by game so they can never bleed
        into another game's state."""
        gid = getattr(event, "game_instance_id", None)
        if gid is None:
            return
        self._state(gid).apply_event(event)

    def act(self, turn: Any) -> dict:
        """Decide and return {"action": <wire action>, "reasoning": <brief>} for one turn."""
        t0 = time.perf_counter()
        state = self._state(turn.game_instance_id)
        if state.seat.seat is None and getattr(turn, "seat", None) is not None:
            state.seat.seat = int(turn.seat)
        action_kind = turn.action_kind
        legal = turn.legal_action or {}

        remaining = self._remaining_s(turn.deadline_at)
        # Budget guard: not enough time for even one model call + submission -> deterministic fallback.
        if remaining is not None and remaining <= self.config.submit_margin_s:
            return self._fallback(state, turn, t0, repair_attempted=False, error_type="deadline_guard",
                                  diag=_StructuredDiag(False))

        messages = build_messages(state, action_kind, legal, turn.phase, turn.deadline_at,
                                  system_prompt_text=self.system_prompt_text)
        response_format = self._response_format(action_kind, legal)
        structured_requested = response_format is not None
        diag = _StructuredDiag(requested=structured_requested)
        call_timeout = self._call_timeout(remaining)

        result = self.brain.complete(messages, response_format=response_format, timeout=call_timeout)
        if result.structured_rejected:
            # Ladder rung: structured output rejected -> retry once without it, same budget. We never
            # label the request "enforced": json_schema strict=false is best-effort guidance only.
            diag.bypassed("provider_rejected_response_format")
            result = self.brain.complete(messages, response_format=None,
                                         timeout=self._call_timeout(self._remaining_s(turn.deadline_at)))

        decision = interpret(action_kind, legal, result.content if result.ok else "")
        if decision.legal:
            return self._commit(state, turn, t0, decision, result,
                                repair_attempted=False, fallback_used=False, diag=diag)

        # Ladder rung: one repair call, only if there is a documented safety margin of time left. The
        # repair prompt gets the EXACT validation reason + required envelope + legal-action schema (no
        # private content) so the model can correct the specific shape error.
        if self._can_repair(turn.deadline_at):
            repair_msgs = list(messages) + [
                {"role": "assistant", "content": result.content or ""},
                {"role": "user", "content": _repair_message(action_kind, legal, decision.failure_reason)},
            ]
            r2 = self.brain.complete(repair_msgs, response_format=response_format,
                                     timeout=self._call_timeout(self._remaining_s(turn.deadline_at)))
            if r2.structured_rejected:
                diag.bypassed("provider_rejected_response_format")
                r2 = self.brain.complete(repair_msgs, response_format=None,
                                         timeout=self._call_timeout(self._remaining_s(turn.deadline_at)))
            decision2 = interpret(action_kind, legal, r2.content if r2.ok else "")
            if decision2.legal:
                return self._commit(state, turn, t0, decision2, r2,
                                    repair_attempted=True, fallback_used=False, diag=diag,
                                    initial=decision)
            return self._fallback(state, turn, t0, repair_attempted=True,
                                  error_type=r2.error_type or "invalid_after_repair", brain=r2,
                                  carry_update=decision2.state_update or decision.state_update,
                                  diag=diag, initial=decision, repair=decision2)

        return self._fallback(state, turn, t0, repair_attempted=False,
                              error_type=result.error_type or "invalid_no_repair_budget", brain=result,
                              carry_update=decision.state_update, diag=diag, initial=decision)

    # ---- ladder helpers ------------------------------------------------------------------------
    def _commit(self, state: GameState, turn: Any, t0: float, decision, result: BrainResult,
                *, repair_attempted: bool, fallback_used: bool, diag: "_StructuredDiag",
                initial=None) -> dict:
        state.apply_state_update(decision.state_update)
        state.previous_action = decision.action
        state.turns_taken += 1
        if repair_attempted:
            state.repair_count += 1
        self._telemetry(turn, t0, result, repair_attempted=repair_attempted,
                        fallback_used=fallback_used, action_valid=True, error_type=None,
                        diag=diag, failure_category=None, fallback_action=None,
                        initial=initial, repair=None)
        return {"action": decision.action, "reasoning": decision.brief_reasoning}

    def _fallback(self, state: GameState, turn: Any, t0: float, *, repair_attempted: bool,
                  error_type: str | None, brain: BrainResult | None = None,
                  carry_update: dict | None = None, diag: "_StructuredDiag" | None = None,
                  initial=None, repair=None) -> dict:
        if carry_update:
            state.apply_state_update(carry_update)
        action = fallback_action(state, turn.action_kind, turn.legal_action or {})
        state.previous_action = action
        state.turns_taken += 1
        state.fallback_count += 1
        if repair_attempted:
            state.repair_count += 1
        self._telemetry(turn, t0, brain, repair_attempted=repair_attempted,
                        fallback_used=True, action_valid=True, error_type=error_type,
                        diag=diag or _StructuredDiag(False),
                        failure_category=_failure_category(error_type),
                        fallback_action=action, initial=initial, repair=repair)
        return {"action": action, "reasoning": "(deterministic legal fallback)"}

    def _response_format(self, action_kind: str, legal_action: dict) -> dict | None:
        mode = self.config.structured_output
        if mode == "off":
            return None
        provider = self.config.model.split("/", 1)[0].strip().lower()
        action_schema = (legal_action or {}).get("schema")
        if mode == "auto":
            if provider not in _STRUCTURED_SCHEMA_PROVIDERS:
                return None
            mode = "json_schema" if isinstance(action_schema, dict) else "json_object"
        if mode == "json_schema" and isinstance(action_schema, dict):
            name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in action_kind)[:60] or "act"
            return {"type": "json_schema", "json_schema": {
                "name": name, "strict": False, "schema": response_schema(action_kind, legal_action)}}
        if mode in {"json_object", "json_schema"}:
            return {"type": "json_object"}
        return None

    # ---- deadline math -------------------------------------------------------------------------
    def _remaining_s(self, deadline_at: str | None) -> float | None:
        dl = _parse_deadline(deadline_at)
        if dl is None:
            return None
        return (dl - self._now()).total_seconds()

    def _call_timeout(self, remaining: float | None) -> float:
        if remaining is None:
            return self.config.brain_timeout_s
        return max(1.0, min(self.config.brain_timeout_s, remaining - self.config.submit_margin_s))

    def _can_repair(self, deadline_at: str | None) -> bool:
        remaining = self._remaining_s(deadline_at)
        if remaining is None:
            return True
        # Need room for a full brain call plus the submission margin before starting a repair.
        return remaining > (self.config.brain_timeout_s + self.config.submit_margin_s) or \
            remaining > self.config.submit_margin_s * 2

    # ---- telemetry (privacy-preserving) --------------------------------------------------------
    def _telemetry(self, turn: Any, t0: float, brain: BrainResult | None, *,
                   repair_attempted: bool, fallback_used: bool, action_valid: bool,
                   error_type: str | None, diag: "_StructuredDiag | None" = None,
                   failure_category: str | None = None, fallback_action: dict | None = None,
                   initial=None, repair=None) -> None:
        if not self.config.log_path:
            return
        diag = diag or _StructuredDiag(False)
        record = {
            "timestamp": self._now().isoformat(),
            "agent_name": self.agent_name,
            "policy_version": self.policy_version,
            "prompt_hash": self.prompt_hash_value,
            "run_id": self.run_id,
            "game_id": getattr(turn, "game_instance_id", None),
            "turn_id": getattr(turn, "turn_id", None),
            "seat": getattr(turn, "seat", None),
            "phase": getattr(turn, "phase", None),
            "action_kind": getattr(turn, "action_kind", None),
            "requested_model": self.config.model,
            "resolved_model": getattr(brain, "resolved_model", None) if brain else None,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": getattr(brain, "prompt_tokens", None) if brain else None,
            "completion_tokens": getattr(brain, "completion_tokens", None) if brain else None,
            "repair_attempted": repair_attempted,
            "fallback_used": fallback_used,
            "action_valid": action_valid,
            "error_type": error_type,
            # Diagnostic fields (safe metadata only) so a future failure is debuggable without a rerun:
            "structured_output_mode": self.config.structured_output,
            "structured_output_requested": diag.requested,
            "structured_output_attempted": diag.requested,
            # Honest: json_schema strict=false is best-effort guidance; we never claim enforcement.
            "structured_output_enforced": diag.status,
            "structured_output_fallback_reason": diag.fallback_reason,
            "validation_failure_category": failure_category,
            "validation_failure_detail": (getattr(repair, "failure_reason", None)
                                          or getattr(initial, "failure_reason", None)),
            "fallback_action_type": _fallback_action_type(fallback_action) if fallback_used else None,
            # Safe STRUCTURAL fingerprints (types/field-names/lengths only, never content):
            "initial_shape": getattr(initial, "fingerprint", None),
            "repair_shape": getattr(repair, "fingerprint", None),
        }
        # Never logged: API keys, bearer tokens, credential files, full private night observations,
        # full prompts, provider reasoning, or any other game's state. Only safe metadata is written.
        try:
            with open(self.config.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass


def _failure_category(error_type: str | None) -> str | None:
    """Coarse, safe classification of why a turn fell back (no raw output retained)."""
    if not error_type:
        return None
    et = error_type.lower()
    if "deadline" in et:
        return "deadline"
    if "structured" in et:
        return "structured_rejected"
    if "invalid_after_repair" in et or "invalid_no_repair_budget" in et:
        return "action_shape_invalid"
    return "model_error"


def _fallback_action_type(action: dict | None) -> str | None:
    """Coarse type of the deterministic fallback action chosen (no game content)."""
    if not isinstance(action, dict):
        return None
    if action.get("pass") is True:
        return "pass"
    if action.get("mode") == "center":
        return "center_inspect"
    if "index" in action:
        return "center_swap"
    if action.get("a") is None and action.get("b") is None and ("a" in action or "b" in action):
        return "decline_swap"
    if "target" in action:
        t = action.get("target")
        if t is None:
            return "decline"
        if t in (-1, NO_ONE_REF):
            return "no_one"
        return "target"
    return "other"


class _StructuredDiag:
    """Tracks, honestly, whether structured output was requested and whether it was bypassed. We do
    not claim a provider ENFORCED the schema (json_schema strict=false is best-effort), only what the
    client attempted and observed."""

    def __init__(self, requested: bool):
        self.requested = requested
        self._bypassed = False
        self.fallback_reason: str | None = None

    def bypassed(self, reason: str) -> None:
        self._bypassed = True
        self.fallback_reason = reason

    @property
    def status(self) -> str:
        if not self.requested:
            return "off"
        return "bypassed" if self._bypassed else "best_effort"


def _repair_message(action_kind: str, legal_action: dict, reason: str | None) -> str:
    schema = (legal_action or {}).get("schema") or {}
    why = f" The exact validation error was: {reason}." if reason else ""
    return (
        "Your previous reply's action was INVALID for this turn." + why +
        " Reply again with ONLY one JSON object of the form "
        '{"action": <the action>, "brief_reasoning": "<short>", "state_update": {...}}. '
        "The \"action\" field must match this JSON schema EXACTLY (use these exact keys, no extra "
        f"fields, no \"type\"/\"text\" wrappers):\n{json.dumps(schema, sort_keys=True)}"
    )


# Module-level handlers so `arena-agent play ... examples/wolfforge_v2_agent.py` works. Run id and
# model come from env (WOLFFORGE_V2_*); the SDK supplies events and turns.
_default = WolfForgeV2Agent(run_id=os.environ.get("WOLFFORGE_V2_RUN_ID"))
on_event = _default.on_event
act = _default.act
