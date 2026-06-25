"""Central config: loads the OpenRouter key from .env and exposes caps + the agent roster.

The API key is read once here and never printed or logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.environ.get(
    "ARENA_STORE_DIR",
    "/tmp/persuasion-arena-store" if os.environ.get("VERCEL") else str(ROOT / "store"),
))

if not os.environ.get("VERCEL"):
    load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
STRUCTURED_OUTPUT_MODES = {"off", "auto", "json_object", "json_schema"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_reasoning_effort(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in REASONING_EFFORTS else default


def _env_structured_output(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in STRUCTURED_OUTPUT_MODES else default


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Put it in a .env file at the repo root."
        )
    return key


def has_api_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def set_api_key(key: str) -> None:
    """Set the key for this process and persist it to the local .env (local-dev convenience).

    The value is never printed or logged. .env is gitignored.
    """
    key = key.strip()
    os.environ["OPENROUTER_API_KEY"] = key
    env_path = ROOT / ".env"
    lines, found = [], False
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                lines.append(f"OPENROUTER_API_KEY={key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"OPENROUTER_API_KEY={key}")
    env_path.write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class Caps:
    """Bounds that keep a game finite and cheap."""
    discussion_rounds: int = _env_int("ARENA_DISCUSSION_ROUNDS", 5)
    max_tokens_per_turn: int = _env_int("ARENA_MAX_TOKENS_PER_TURN", 4000)
    request_timeout_s: float = _env_float("ARENA_REQUEST_TIMEOUT_S", 60.0)
    retries: int = _env_int("ARENA_RETRIES", 1)
    temperature: float = _env_float("ARENA_TEMPERATURE", 0.8)
    reasoning_effort: str = _env_reasoning_effort("ARENA_REASONING_EFFORT", "medium")
    prior_message_turns: int = _env_int("ARENA_PRIOR_MESSAGE_TURNS", 8)
    openrouter_structured_output: str = _env_structured_output(
        "ARENA_OPENROUTER_STRUCTURED_OUTPUT",
        "off",
    )


def validate_caps(caps: Caps) -> Caps:
    if caps.discussion_rounds <= 0:
        raise ValueError("discussion_rounds must be positive")
    if caps.max_tokens_per_turn <= 0:
        raise ValueError("max_tokens_per_turn must be positive")
    if caps.request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")
    if caps.retries < 0:
        raise ValueError("retries must be nonnegative")
    if caps.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if caps.reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    if caps.prior_message_turns < 0:
        raise ValueError("prior_message_turns must be nonnegative")
    return caps


def caps_with_overrides(base: Caps | None = None, **overrides) -> Caps:
    values = {k: v for k, v in overrides.items() if v is not None}
    if "reasoning_effort" in values:
        values["reasoning_effort"] = str(values["reasoning_effort"]).strip().lower()
    return validate_caps(replace(base or SETTINGS.caps, **values))


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    harness: str = "base"


@dataclass
class Settings:
    caps: Caps = field(default_factory=Caps)

    def roster(self) -> list[AgentSpec]:
        configured = Path(os.environ.get("ARENA_AGENTS_FILE", "agents.yaml"))
        roster_path = configured if configured.is_absolute() else ROOT / configured
        with open(roster_path) as f:
            data = yaml.safe_load(f)
        return [AgentSpec(**a) for a in data["agents"]]


SETTINGS = Settings()
