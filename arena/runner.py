"""Play games and persist them. A run = N games with a fixed agent roster.

Thin wrapper around `batch.run_batch` so there is exactly ONE scheduling / role-balancing /
round-cap implementation feeding the store. Kept for the documented single-game command
(`play_run('onuw', n_games=1, seed_base=<seed>)`); it simply runs the batch with one worker.
"""
from __future__ import annotations

from .config import SETTINGS
from .batch import run_batch


def play_run(game: str = "onuw", n_games: int = 1, seed_base: int = 8841,
             run_id: str | None = None, roster=None) -> str:
    return run_batch(game=game, n_games=n_games, seed_base=seed_base, run_id=run_id,
                     roster=roster, workers=1,
                     discussion_rounds=SETTINGS.caps.discussion_rounds)
