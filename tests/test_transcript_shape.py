"""Every game core returns its transcript through arena.games.base.Game._transcript(), so the
top-level shape can't silently drift between ONUW / Avalon / Secret Mafia. Pins the shared
top-level key set and the per-seat record shape across all three."""
import pytest

from arena.games.avalon import Avalon
from arena.games.onuw import ONUW
from arena.games.secret_mafia import SecretMafia
from tests.scripted import ScriptedDefault

CORE_KEYS = {"game", "title", "seed", "meta", "players", "agentCallLog",
             "cardsInPlay", "center", "phases", "outcome", "winner_team"}
SEAT_KEYS = {"seat", "dealt", "end", "team", "believes", "won"}


@pytest.mark.parametrize("cls", [ONUW, Avalon, SecretMafia])
def test_transcript_core_shape(cls):
    names = {i: f"P{i}" for i in range(5)}
    rec = cls(names, seed=11).play({i: ScriptedDefault() for i in range(5)})
    assert CORE_KEYS <= set(rec), f"{cls.__name__} missing top-level keys: {CORE_KEYS - set(rec)}"
    assert rec["game"] == cls.GAME and rec["title"] == cls.TITLE
    assert rec["outcome"] == rec["phases"][-1]["outcome"]
    assert rec["winner_team"] == rec["outcome"]["team"]
    for p in rec["players"]:
        assert SEAT_KEYS <= set(p), f"{cls.__name__} per-seat record missing: {SEAT_KEYS - set(p)}"
