"""Pure event-sourcing conformance.

Each seat's event stream (public events + its own private events) must reconstruct that seat's
legitimate filtered view exactly. If this passes, the connected turn can be slimmed to the action
request only: a stateful agent rebuilds all context from the deltas it polls.

The strongest assertion here is `SeatState.render_context() == ONUW.base_prompt(seat)` byte-for-byte
at end of game — the reconstructed prompt is identical to what the engine used to spoon-feed.
"""
from __future__ import annotations

import pytest

from arena.games.onuw import ONUW
from examples.seat_state import SeatState
from tests.scripted import ScriptedDefault


def _play_capturing(names, seed, *, deck=None, deck_preset="arena", deal_override=None, rounds=2):
    events = []

    def sink(etype, payload, *, phase=None, visibility="public", target_seat=None):
        events.append({"type": etype, "payload": payload, "phase": phase,
                       "visibility": visibility, "target_seat": target_seat})

    game = ONUW(names, seed=seed, discussion_rounds=rounds, deck=deck, deck_preset=deck_preset,
                deal_override=deal_override, event_sink=sink)
    game.play({i: ScriptedDefault() for i in range(len(names))})
    return game, events


def _seat_events(events, seat):
    # exactly what a connected poll delivers to a seat: public events + that seat's private events
    return [e for e in events if e["visibility"] == "public" or e["target_seat"] == seat]


def _assert_reconstructs(game, events):
    for seat in range(game.n):
        st = SeatState(seat)
        for e in _seat_events(events, seat):
            st.apply(e)
        assert st.night_obs == game.obs[seat], f"seat {seat}: night obs mismatch"
        assert st.believed_role == game.believes[seat], f"seat {seat}: believes mismatch"
        assert st.deck_line() == game._deck_line(), f"seat {seat}: deck line mismatch"
        assert st.roster_line() == game._roster_line(), f"seat {seat}: roster line mismatch"
        # the whole filtered prompt, byte-for-byte — the real completeness guarantee
        assert st.render_context() == game.base_prompt(seat), f"seat {seat}: context mismatch"


@pytest.mark.parametrize("preset", ["arena", "classic", "tanner"])
@pytest.mark.parametrize("n", [5, 6, 7])
def test_reconstruct_across_decks_and_counts(preset, n):
    names = {i: f"P{i}" for i in range(n)}
    played = 0
    for seed in range(8000, 8016):
        try:
            game, events = _play_capturing(names, seed, deck_preset=preset)
        except ValueError:
            pytest.skip(f"{preset} does not support {n} players")
        _assert_reconstructs(game, events)
        played += 1
    assert played > 0


def test_slim_turn_carries_no_context_but_events_reconstruct():
    """Under delta transport the turn carries NO context (base_prompt == ""), yet a stateful harness
    rebuilds the full view from the event stream. A stateless harness would see only the empty
    context + the action instruction — i.e. play blind. This is the contract's teeth."""
    names = {i: f"P{i}" for i in range(5)}
    game, events = _play_capturing(names, seed=8001, deck_preset="arena")  # captured with event_driven via flag below
    # replay the same game with delta transport on, to confirm slim turns
    slim_events = []

    def sink(etype, payload, *, phase=None, visibility="public", target_seat=None):
        slim_events.append({"type": etype, "payload": payload, "phase": phase,
                            "visibility": visibility, "target_seat": target_seat})

    slim = ONUW(names, seed=8001, discussion_rounds=2, deck_preset="arena",
                event_sink=sink, event_driven=True)
    slim.play({i: ScriptedDefault() for i in range(5)})

    # 1) slim: the per-turn prompt carries no context at all
    for seat in range(5):
        assert slim.base_prompt(seat) == "", f"seat {seat}: expected slim (empty) context"

    # 2) but the event stream still reconstructs each seat's FULL view
    slim.event_driven = False  # toggle off only to read the reference full view
    for seat in range(5):
        st = SeatState(seat)
        for e in _seat_events(slim_events, seat):
            st.apply(e)
        assert st.render_context() == slim.base_prompt(seat), f"seat {seat}: slim reconstruction mismatch"
        assert st.render_context().strip() != "", f"seat {seat}: reconstruction should be non-trivial"


def test_reconstruct_rich_night_layout():
    # Force the roles the standard presets rarely/never deal (Doppelganger, Masons) so every night
    # path is event-sourced: 5 players + 3 center.
    names = {i: f"P{i}" for i in range(5)}
    layout = ["Doppelganger", "Mason", "Mason", "Werewolf", "Seer",   # players
              "Robber", "Troublemaker", "Insomniac"]                  # center
    for seed in range(500, 512):
        game, events = _play_capturing(names, seed, deck=list(layout), deal_override=list(layout))
        _assert_reconstructs(game, events)
