from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .agent import ArenaAgent
from .credentials import CredentialsStore, DEFAULT_SERVER, redact_token


def _load_harness(path: str):
    """Load a harness module from a file. It MUST define act(turn); it MAY define on_event(event)
    to build state from the delta event stream — required for stateful play under delta transport."""
    p = Path(path)
    spec = importlib.util.spec_from_file_location("arena_user_agent", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agent file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "act"):
        raise RuntimeError("agent file must define act(turn)")
    return mod


def _cmd_play(args) -> int:
    agent = ArenaAgent(name=args.name, server=args.server,
                       model=args.model, harness=args.harness,
                       credentials=CredentialsStore(args.credentials) if args.credentials else None)
    mod = _load_harness(args.file)
    agent.act(mod.act)
    if hasattr(mod, "on_event"):
        # Deliver the delta event stream so a stateful harness can build its own memory. Without
        # this the harness sees only its turns and plays blind under delta transport.
        agent.on_event(mod.on_event)
    signup = agent.signup(run_id=args.run, game=args.game)
    print(json.dumps({"signup_id": signup.signup_id, "run_id": signup.run_id,
                      "status": signup.status, "seat": signup.seat}, sort_keys=True))
    if args.once:
        agent.run_once([signup])
    else:
        agent.run_forever([signup])
    return 0


def _cmd_status(args) -> int:
    store = CredentialsStore(args.credentials) if args.credentials else CredentialsStore()
    creds = store.get(args.server.rstrip("/"))
    if not creds:
        print(json.dumps({"configured": False, "server": args.server}, sort_keys=True))
        return 1
    print(json.dumps({"configured": True, **creds.redacted()}, sort_keys=True))
    return 0


def _cmd_credentials(args) -> int:
    store = CredentialsStore(args.credentials) if args.credentials else CredentialsStore()
    data = store.load_all()
    redacted = {"version": data.get("version", 1), "profiles": {}}
    for server, profile in data.get("profiles", {}).items():
        redacted["profiles"][server] = {
            **profile,
            "agent_token": redact_token(profile.get("agent_token")),
        }
    print(json.dumps(redacted, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arena-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    play = sub.add_parser("play", help="connect a local agent harness to a run")
    target = play.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", help="concrete run id to join")
    target.add_argument("--game", help="discover and join the first open run for a game")
    play.add_argument("--server", default=DEFAULT_SERVER)
    play.add_argument("--name", default="local-agent")
    play.add_argument("--model", help="declared model identity for leaderboard metadata")
    play.add_argument("--harness", help="declared harness identity for leaderboard metadata")
    play.add_argument("--credentials", help="credential store path for this local agent identity")
    play.add_argument("--once", action="store_true", help="perform one SDK loop step and exit")
    play.add_argument("file", help="Python file defining act(turn)")
    play.set_defaults(func=_cmd_play)

    status = sub.add_parser("status", help="show configured agent credentials")
    status.add_argument("--server", default=DEFAULT_SERVER)
    status.add_argument("--credentials", help="credential store path")
    status.set_defaults(func=_cmd_status)

    creds = sub.add_parser("credentials", help="print local credential profiles")
    creds.add_argument("--credentials", help="credential store path")
    creds.set_defaults(func=_cmd_credentials)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"arena-agent: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
