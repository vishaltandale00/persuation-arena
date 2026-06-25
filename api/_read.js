// Shared read-side helpers for the Vercel JS port — pure logic ported from the Python arena
// (arena/score.py, arena/server.py, arena/store.py, arena/games/onuw.py, arena/config.py).
// Everything here is read-only / derived: scoring, event shaping for the API, the ONUW deck
// presets, and the OpenRouter model catalog. All DB access goes through q() from ./_db.js.
import { q } from './_db.js';

// --- round-to-3 matching Python round() -----------------------------------------------------------
// Python's round() is banker's rounding, but the codebase's intent here is plain 3-decimal display;
// Math.round(x*1000)/1000 reproduces it for the values we produce (rates/CI bounds in [0,1]).
const r3 = (x) => Math.round(x * 1000) / 1000;

// === 1. scoreRun — port of arena/score.py ========================================================

/** Wilson score interval. Returns {lo, hi, p} for k successes in n trials (p = k/n). */
export function wilson(k, n, z = 1.96) {
  if (n === 0) return { lo: 0.0, hi: 0.0, p: 0.0 };
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return { lo: Math.max(0.0, center - half), hi: Math.min(1.0, center + half), p };
}

/** A scorecard cell {w, n, rate, lo, hi} with rate/lo/hi rounded to 3 decimals. */
function _cell(w, n) {
  const { lo, hi, p } = wilson(w, n);
  return { w, n, rate: r3(p), lo: r3(lo), hi: r3(hi) };
}

/**
 * Port of arena/games/base.py is_no_contest: a game with no evil seat AND no winning seat is the
 * ONUW degenerate (no Werewolf/Minion in play, the vote killed an innocent) — no opposing faction,
 * so it is dropped from win-rate scoring. A no-wolf game the village idles to a win keeps its winners.
 */
function isNoContest(seats) {
  const noEvil = !seats.some((s) => s.team === 'evil');
  const noWinner = !seats.some((s) => (Number(s.won) || 0) > 0);
  return noEvil && noWinner;
}

/**
 * agent -> scorecard. Reads game_players rows for the run and aggregates per agent:
 *   {overall, good, evil} cells, by_role:{dealt_role: cell}, and {calls, forfeits, forfeit_rate}.
 * Team bucket is r.team if in ('good','evil') else 'good'; dealt_role falls back to '?'.
 */
export async function scoreRun(runId) {
  const rows = await q(
    `SELECT gid, agent, team, won, dealt_role, end_role, calls, forfeits
       FROM game_players WHERE run_id = $1`,
    [runId],
  );

  // group by game so no-contest games (no evil seat, no winner) drop out of scoring
  const games = {};
  for (const r of rows) (games[r.gid] ||= []).push(r);

  const agg = {}; // agent -> aggregate accumulator
  for (const seats of Object.values(games)) {
    if (isNoContest(seats)) continue;
    for (const r of seats) {
      let a = agg[r.agent];
      if (!a) {
        a = agg[r.agent] = {
          overall: [0, 0], good: [0, 0], evil: [0, 0],
          by_role: {}, calls: 0, forfeits: 0,
        };
      }
      const won = Number(r.won) || 0;
      a.overall[1] += 1;
      a.overall[0] += won;
      const t = (r.team === 'good' || r.team === 'evil') ? r.team : 'good';
      a[t][1] += 1;
      a[t][0] += won;
      const role = r.dealt_role || '?';
      const cell = (a.by_role[role] ||= [0, 0]);
      cell[1] += 1;
      cell[0] += won;
      a.calls += Number(r.calls) || 0;
      a.forfeits += Number(r.forfeits) || 0;
    }
  }

  const out = {};
  for (const [agent, d] of Object.entries(agg)) {
    const { calls, forfeits } = d;
    const by_role = {};
    // sorted by role to mirror Python's sorted(d["by_role"].items())
    for (const role of Object.keys(d.by_role).sort()) {
      const [w, n] = d.by_role[role];
      by_role[role] = _cell(w, n);
    }
    out[agent] = {
      overall: _cell(d.overall[0], d.overall[1]),
      good: _cell(d.good[0], d.good[1]),
      evil: _cell(d.evil[0], d.evil[1]),
      by_role,
      calls,
      forfeits,
      forfeit_rate: calls ? r3(forfeits / calls) : 0.0,
    };
  }
  return out;
}

// === 2. apiEvent — port of arena/server.py _api_event ===========================================

/**
 * Shape a run_events row for the API. Mirrors store.list_run_events: the stored row's payload lives
 * in a TEXT column `payload_json` parsed to an object and exposed as `payload`, and the row id is
 * exposed as `event_id`. This accepts a row already shaped that way (event_id, seq, visibility,
 * phase, type, payload, game_instance_id); if payload is still a JSON string it is parsed here so
 * actor_seat extraction matches the Python isinstance(payload, dict) check.
 */
export function apiEvent(eventRow) {
  let payload = eventRow.payload;
  if (payload == null && typeof eventRow.payload_json !== 'undefined') {
    payload = eventRow.payload_json; // tolerate a not-yet-renamed raw row
  }
  if (typeof payload === 'string') {
    try { payload = JSON.parse(payload); } catch { /* leave as string */ }
  }
  const isObj = payload !== null && typeof payload === 'object' && !Array.isArray(payload);
  return {
    event_id: eventRow.event_id ?? eventRow.id,
    game_instance_id: eventRow.game_instance_id ?? null,
    seq: eventRow.seq,
    visibility: eventRow.visibility,
    phase: eventRow.phase ?? null,
    type: eventRow.type,
    actor_seat: isObj ? (payload.actor_seat ?? null) : null,
    payload,
  };
}

// === 3. deckForApi — port of arena/server.py _deck_for_api + arena/games/onuw.py decks ===========

export const DEFAULT_DECK_PRESET = 'arena';

// Ported faithfully from arena/games/onuw.py _DECK_PRESETS. Each preset defines an explicit deck
// per table size (5/6/7), exactly (n_players + 3) cards so 3 stay in the center. label/description
// are retained for parity though deckForApi only uses the deck.
const _DECK_PRESETS = {
  arena: {
    label: 'Arena pressure',
    description: 'Verifiable Mason pair as a trust nucleus + Tanner; a third werewolf and the Drunk join as the table grows.',
    decks: {
      5: ['Werewolf', 'Werewolf', 'Minion', 'Seer', 'Tanner', 'Robber', 'Mason', 'Mason'],
      6: ['Werewolf', 'Werewolf', 'Minion', 'Seer', 'Tanner', 'Robber', 'Drunk', 'Mason', 'Mason'],
      7: ['Werewolf', 'Werewolf', 'Werewolf', 'Minion', 'Seer', 'Tanner', 'Robber', 'Drunk', 'Mason', 'Mason'],
    },
  },
  classic: {
    label: 'Classic',
    description: 'Original simple scaffold: wolves, Minion, core information roles, Villager cover.',
    decks: {
      5: ['Werewolf', 'Werewolf', 'Seer', 'Robber', 'Troublemaker', 'Minion', 'Villager', 'Villager'],
      6: ['Werewolf', 'Werewolf', 'Seer', 'Robber', 'Troublemaker', 'Minion', 'Villager', 'Villager', 'Villager'],
      7: ['Werewolf', 'Werewolf', 'Seer', 'Robber', 'Troublemaker', 'Minion', 'Villager', 'Villager', 'Villager', 'Villager'],
    },
  },
  tanner: {
    label: 'Tanner puzzle',
    description: 'High-uncertainty Tanner/Drunk/Insomniac setup; Minion/Hunter join larger tables.',
    decks: {
      5: ['Werewolf', 'Werewolf', 'Troublemaker', 'Robber', 'Insomniac', 'Drunk', 'Seer', 'Tanner'],
      6: ['Werewolf', 'Werewolf', 'Troublemaker', 'Robber', 'Insomniac', 'Drunk', 'Seer', 'Tanner', 'Minion'],
      7: ['Werewolf', 'Werewolf', 'Troublemaker', 'Robber', 'Insomniac', 'Drunk', 'Seer', 'Tanner', 'Minion', 'Hunter'],
    },
  },
};

/** Port of onuw.normalize_deck_preset: lowercase, spaces->underscores; throws on unknown. */
export function normalizeDeckPreset(preset) {
  const key = (preset || DEFAULT_DECK_PRESET).trim().toLowerCase().replace(/ /g, '_');
  if (!(key in _DECK_PRESETS)) {
    throw new Error(`unknown ONUW deck preset: ${preset}`);
  }
  return key;
}

/** Port of onuw.deck_for_preset: the preset's explicit deck for this table size. */
export function deckForPreset(nPlayers, preset = null) {
  const key = normalizeDeckPreset(preset);
  const deck = _DECK_PRESETS[key].decks[nPlayers];
  if (!deck) {
    throw new Error(`ONUW deck preset ${key} does not support ${nPlayers} players`);
  }
  return deck.slice();
}

/** Port of server._deck_for_api: null unless game === 'onuw', else the preset's deck for this size.
 * An out-of-range player count (e.g. a partially-configured run) degrades to null, not a throw. */
export function deckForApi(game, players, deckPreset) {
  if (game !== 'onuw') return null;
  try {
    return deckForPreset(players, deckPreset || DEFAULT_DECK_PRESET);
  } catch {
    return null;
  }
}

// === 4. fetchModels — port of arena/server.py _fetch_models + arena/config.py ====================

export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';

// Module-level ~10 min cache mirroring Python's _models_cache.
const _models_cache = { at: 0, data: [] };

/**
 * OpenRouter's model catalog as [{id, name}] sorted by id.toLowerCase(), cached ~10 min.
 * The /models endpoint is public, so the Authorization header is optional. On any failure we
 * return the last good cache (or [] on the very first miss) so the picker degrades gracefully.
 */
export async function fetchModels() {
  const now = Date.now() / 1000; // seconds, to match Python's time.time()
  if (_models_cache.data.length && now - _models_cache.at < 600) {
    return _models_cache.data;
  }
  const headers = {};
  if (process.env.OPENROUTER_API_KEY) {
    headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
  }
  try {
    const r = await fetch(OPENROUTER_BASE_URL + '/models', { headers });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    const models = (body.data || [])
      .filter((m) => m && m.id)
      .map((m) => ({ id: m.id, name: m.name || m.id }));
    models.sort((a, b) => {
      const x = a.id.toLowerCase(), y = b.id.toLowerCase();
      return x < y ? -1 : x > y ? 1 : 0;
    });
    _models_cache.at = now;
    _models_cache.data = models;
    return models;
  } catch {
    return _models_cache.data;
  }
}

// === 5. ROSTER_FALLBACK — known competitor models so the picker is never blank ===================

export const ROSTER_FALLBACK = [
  { id: 'openai/gpt-5.5', name: 'OpenAI: GPT-5.5' },
  { id: 'x-ai/grok-4.20', name: 'xAI: Grok 4.20' },
  { id: 'deepseek/deepseek-v4-pro', name: 'DeepSeek: DeepSeek V4 Pro' },
  { id: 'anthropic/claude-opus-4.8', name: 'Anthropic: Claude Opus 4.8' },
  { id: 'google/gemini-3.1-pro-preview', name: 'Google: Gemini 3.1 Pro Preview' },
];

/** Just the ids of ROSTER_FALLBACK. */
export function rosterModels() {
  return ROSTER_FALLBACK.map((m) => m.id);
}
