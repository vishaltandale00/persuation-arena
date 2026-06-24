from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from .client import ArenaHttpClient
from .credentials import AgentCredentials, CredentialsStore, DEFAULT_SERVER
from .models import Event, PollResponse, Signup, Turn


ActHandler = Callable[[Turn], dict[str, Any]]
EventHandler = Callable[[Event], None]


class ArenaAgent:
    def __init__(self, name: str, server: str = DEFAULT_SERVER,
                 credentials: CredentialsStore | None = None,
                 client: ArenaHttpClient | None = None):
        self.name = name
        self.server = server.rstrip("/")
        self.credentials = credentials or CredentialsStore()
        self.client = client or ArenaHttpClient(self.server)
        # Registration + signup + status stay on the central API (self.client). Once the run's per-run
        # coordinator container publishes its tunnel URL, ready/poll/reply switch to it (self._coord_client).
        self._coord_client: ArenaHttpClient | None = None
        self._act_handler: ActHandler | None = None
        self._event_handler: EventHandler | None = None
        self._last_event_by_signup: dict[str, str | None] = {}

    def act(self, fn: ActHandler) -> ActHandler:
        self._act_handler = fn
        return fn

    def on_event(self, fn: EventHandler) -> EventHandler:
        """Register a handler called for every new event the agent polls, in order, BEFORE any turn
        in the same poll. The SDK only *delivers* events (it holds nothing but the cursor) — building
        state from them, or not, is the harness's job. A harness that ignores them stays stateless
        and, under delta transport, plays blind."""
        self._event_handler = fn
        return fn

    def ensure_registered(self) -> AgentCredentials:
        existing = self.credentials.get(self.server)
        if existing:
            return existing
        creds = self.client.register_agent(self.name)
        self.credentials.save(creds)
        return creds

    def signup(self, run_id: str | None = None, game: str | None = None,
               max_concurrent_turns: int = 1) -> Signup:
        creds = self.ensure_registered()
        if not run_id:
            runs = self.client.discover_runs(game)
            if not runs:
                raise RuntimeError(f"no open runs found for game={game or '*'}")
            run_id = runs[0]["run_id"]
        signup = self.client.signup_run(creds, run_id, max_concurrent_turns=max_concurrent_turns)
        self._last_event_by_signup.setdefault(signup.signup_id, None)
        self._maybe_repoint(signup)
        return signup

    def _maybe_repoint(self, signup: Signup) -> None:
        """When the run's per-run coordinator publishes its URL, route gameplay (ready/poll/reply) at
        that container. Idempotent; a None URL leaves the agent on the central API."""
        url = getattr(signup, "coordinator_url", None)
        if url and self._coord_client is None:
            self._coord_client = ArenaHttpClient(url)

    def status(self, signup: Signup) -> Signup:
        return self.client.get_signup(self.ensure_registered(), signup.signup_id)

    def step_signup(self, signup: Signup) -> tuple[Signup | None, PollResponse | None]:
        creds = self.ensure_registered()
        # Status from the central API (it carries the coordinator URL); once discovered, gameplay
        # (ready/poll/reply) is routed at the per-run container.
        current = self.client.get_signup(creds, signup.signup_id)
        self._maybe_repoint(current)
        play = self._coord_client or self.client
        if current.status == "ready_required":
            current = play.mark_ready(creds, signup.signup_id)
        if current.status != "active":
            return current, None
        after = self._last_event_by_signup.get(signup.signup_id)
        poll = play.poll_signup(creds, signup.signup_id, after_event_id=after)
        if poll.events:
            # Fold each new event into the agent's state BEFORE it acts, so act() sees everything
            # up to its turn. The SDK owns the cursor; the agent owns the accumulated game state.
            if self._event_handler:
                for ev in poll.events:
                    self._event_handler(ev)
            self._last_event_by_signup[signup.signup_id] = poll.events[-1].event_id
        if poll.turn:
            if not self._act_handler:
                raise RuntimeError("no @agent.act handler registered")
            t0 = time.perf_counter()
            result = self._act_handler(poll.turn)
            client_ms = int((time.perf_counter() - t0) * 1000)
            play.reply_turn(
                creds,
                poll.turn.turn_id,
                result.get("action"),
                result.get("reasoning"),
                result.get("client_ms", client_ms),
            )
        return current, poll

    def run_once(self, signups: list[Signup]) -> list[tuple[Signup | None, PollResponse | None]]:
        return [self.step_signup(s) for s in signups]

    def run_forever(self, signups: list[Signup], max_sleep_ms: int = 5000) -> None:
        active = list(signups)
        while active:
            next_sleep = max_sleep_ms
            keep: list[Signup] = []
            for signup in active:
                try:
                    current, poll = self.step_signup(signup)
                except Exception as e:
                    # A single failed step — the server 422s a rejected action, a transient network
                    # blip, the coordinator briefly unreachable — must NEVER kill the agent. If it did,
                    # one bad turn would forfeit every remaining turn for the whole run. Log it, keep
                    # the signup, and retry on the next poll; the offending turn defaults server-side
                    # at its deadline, after which polling resumes cleanly.
                    print(f"[arena-agent] step error for {signup.signup_id}: {type(e).__name__}: {e}",
                          flush=True)
                    keep.append(signup)
                    continue
                if current and current.status not in {"completed", "rejected", "expired", "cancelled"}:
                    keep.append(current)
                    next_sleep = min(next_sleep, poll.poll_after_ms if poll else current.poll_after_ms)
            active = keep
            if active:
                time.sleep(max(0.05, next_sleep / 1000))
