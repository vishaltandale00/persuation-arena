"""Modal preview deploys must be name/secret isolated from production.

The full-stack preview workflow sets these env vars before `modal deploy`; this test keeps
arena.modal_app from regressing back to hard-coded prod names.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def test_modal_app_names_are_env_configurable():
    env = os.environ.copy()
    env.update(
        ARENA_MODAL_APP_NAME="persuasion-arena-coordinator-pr-123",
        ARENA_MODAL_URL_DICT_NAME="arena-coordinator-urls-pr-123",
        ARENA_MODAL_DB_SECRET_NAME="neon-database-url-pr-123",
        ARENA_MODAL_SPAWN_SECRET_NAME="arena-spawn-token-pr-123",
    )
    script = textwrap.dedent(
        """
        import json
        from arena import modal_app

        print(json.dumps({
            "app": modal_app.APP_NAME,
            "dict": modal_app.URL_DICT_NAME,
            "db_secret": modal_app.DB_SECRET_NAME,
            "spawn_secret": modal_app.SPAWN_SECRET_NAME,
        }))
        """
    )
    out = subprocess.check_output([sys.executable, "-c", script], env=env, text=True)
    assert json.loads(out) == {
        "app": "persuasion-arena-coordinator-pr-123",
        "dict": "arena-coordinator-urls-pr-123",
        "db_secret": "neon-database-url-pr-123",
        "spawn_secret": "arena-spawn-token-pr-123",
    }
