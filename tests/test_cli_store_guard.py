"""The CLI store guard is fail-SAFE: every subcommand is forced onto the local SQLite store unless
it is explicitly in arena.cli._REMOTE_STORE_FUNCS. These tests pin that allowlist and verify the
_dispatch() behavior, so a NEW subcommand can never silently read/write prod Neon (config.load_dotenv
injects .env's DATABASE_URL at import — the exact footgun this guards)."""
import argparse
import os

from arena import cli


def _subcommand_funcs() -> dict:
    parser = cli.build_parser()
    subaction = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {name: sp.get_default("func") for name, sp in subaction.choices.items()}


def test_only_serve_is_remote_every_other_subcommand_is_local():
    funcs = _subcommand_funcs()
    assert funcs, "no subcommands registered"
    assert cli._REMOTE_STORE_FUNCS == {cli._serve}, "remote-store allowlist changed — review for safety"
    for name, func in funcs.items():
        is_remote = func in cli._REMOTE_STORE_FUNCS
        assert is_remote == (name == "serve"), (
            f"subcommand {name!r}: remote={is_remote}; only 'serve' may use the remote store"
        )


def test_dispatch_drops_database_url_for_a_local_command(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://should/be/dropped")
    seen = {}

    def probe(_args):  # not in _REMOTE_STORE_FUNCS -> treated as a local command
        seen["db"] = os.environ.get("DATABASE_URL")

    cli._dispatch(argparse.Namespace(func=probe))
    assert seen["db"] is None, "DATABASE_URL must be dropped before a local command runs"


def test_dispatch_keeps_database_url_for_a_remote_command(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://keep/me")
    seen = {}

    def probe(_args):
        seen["db"] = os.environ.get("DATABASE_URL")

    monkeypatch.setattr(cli, "_REMOTE_STORE_FUNCS", {probe})
    cli._dispatch(argparse.Namespace(func=probe))
    assert seen["db"] == "postgres://keep/me", "an allowlisted command must keep DATABASE_URL"
