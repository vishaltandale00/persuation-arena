// GET /api/runs/{run_id}
// Port of arena/server.py api_run (~line 438), which composes store.get_run (~432) and
// store.list_run_signups (~860), then derives per-agent wins, the team_split, the games list,
// the connected-agent summary, and (when the run has games) the full scorecard via scoreRun.
//
// Shape parity notes (must match the Python dict at server.py:471-485 exactly):
//   - camelCase response keys: nGames, seedBase, deckPreset, teamSplit, connectedSummary.
//   - `wins` is per-agent and null until the run has at least one game.
//   - `scores` is {} until the run has games, then the scoreRun scorecard.
//   - `deckPreset` defaults to DEFAULT_DECK_PRESET for onuw runs with no stored preset, else null.
//   - `deck` is the role list for onuw, else null.
// Column notes (Neon, snake_case; Python reads via SELECT *):
//   - runs.agents_json is a TEXT column holding JSON -> JSON.parse it (neon returns TEXT as string).
//   - players / n_games (INTEGER) and seed_base (BIGINT) are coerced with Number() so the JSON
//     emits plain numbers (BIGINT can arrive as a string from the driver).
//   - game_players.won is INTEGER; SUM(won) (a bigint aggregate) is coerced with Number().
import { q, send, utcnow } from '../../_db.js';
import { scoreRun, deckForApi, DEFAULT_DECK_PRESET, apiEvent } from '../../_read.js';

// Port of store._expire_signup_if_needed (store.py:764-777), which store.list_run_signups
// (store.py:873) applies to EVERY returned signup. Lazily flips a stale 'waiting' signup past its
// waiting_expires_utc, or a stale 'ready_required' signup past its ready_deadline_utc, to 'expired'
// (persisting the write-back), so agents[].status and connectedSummary match Python value-for-value
// instead of returning the raw, not-yet-flipped DB status. Deadlines are ISO-8601 UTC strings in the
// same _utcnow() shape (microsecond precision, trailing Z), so the Python `<` is a lexicographic
// string compare here too (chronological == lexicographic for identical-format timestamps; strict
// `<` means an exactly-equal deadline does NOT expire, matching Python).
async function expireSignupIfNeeded(signup) {
  const now = utcnow();
  let status = signup.status;
  if (status === 'waiting' && signup.waiting_expires_utc && signup.waiting_expires_utc < now) {
    status = 'expired';
  }
  if (status === 'ready_required' && signup.ready_deadline_utc && signup.ready_deadline_utc < now) {
    status = 'expired';
  }
  if (status !== signup.status) {
    await q('UPDATE run_signups SET status = $1, updated_utc = $2 WHERE id = $3',
      [status, now, signup.id]);
    signup.status = status;
    signup.updated_utc = now;
  }
  return signup;
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  const runId = req.query.run_id;

  // store.get_run: load the run row (SELECT *), 404 if missing.
  const r = (await q('SELECT * FROM runs WHERE id = $1', [runId]))[0];
  if (!r) return send(res, 404, { error: 'run not found' });

  // agents_json is TEXT holding JSON -> parse. (May be null/empty for connected-only runs.)
  const agentsSrc = r.agents_json ? JSON.parse(r.agents_json) : [];
  let metadata = {};
  try {
    metadata = r.metadata_json ? JSON.parse(r.metadata_json) : {};
  } catch {
    metadata = {};
  }

  // store.get_run: games for the run, ordered by gid.
  const games = await q(
    'SELECT gid, seed, winner_team, line FROM games WHERE run_id = $1 ORDER BY gid',
    [runId],
  );
  const hasGames = games.length > 0;

  // store.get_run: per-agent wins across the run. Initialise every agent name to 0, then overlay
  // SUM(won) per agent so unseen agents stay 0.
  const wins = {};
  for (const a of agentsSrc) wins[a.name] = 0;
  const wr = await q(
    'SELECT agent, SUM(won) w FROM game_players WHERE run_id = $1 GROUP BY agent',
    [runId],
  );
  for (const row of wr) wins[row.agent] = Number(row.w) || 0;

  // store.get_run: team_split keyed by winner_team, seeded with good/evil at 0.
  const teamSplit = { good: 0, evil: 0 };
  for (const g of games) {
    teamSplit[g.winner_team] = (teamSplit[g.winner_team] || 0) + 1;
  }

  // store.list_run_signups: signups joined to agents, ordered seat-first.
  const signups = await q(
    `SELECT s.*, a.display_name, a.status AS agent_status, a.last_seen_utc
       FROM run_signups s JOIN agents a ON a.id = s.agent_id
      WHERE s.run_id = $1
      ORDER BY s.seat IS NULL, s.seat, s.created_utc`,
    [runId],
  );
  // store.list_run_signups runs _expire_signup_if_needed on every signup, lazily flipping
  // deadline-expired 'waiting'/'ready_required' rows to 'expired' (and persisting). Apply it here so
  // every downstream read of s.status (agents[].status, connectedSummary) sees the effective status.
  for (const s of signups) await expireSignupIfNeeded(s);
  const signupById = {};
  for (const s of signups) signupById[s.id] = s;

  // agents_src: stored roster, or a synthetic one derived from the signups for connected-only runs.
  const builtSrc = agentsSrc.length
    ? agentsSrc
    : signups.map((s) => ({
        name: s.display_name,
        model: 'connected-agent',
        harness: 'connected',
        agent_id: s.agent_id,
        signup_id: s.id,
      }));

  const agents = builtSrc.map((a) => {
    const s = signupById[a.signup_id];
    return {
      name: a.name,
      model: a.model,
      harness: a.harness ?? 'base',
      wins: hasGames ? (wins[a.name] ?? 0) : null,
      agent_id: a.agent_id ?? null,
      signup_id: a.signup_id ?? null,
      status: s ? (s.status ?? null) : null,
      seat: s ? (s.seat ?? null) : null,
      last_poll_utc: s ? (s.last_poll_utc ?? null) : null,
      last_event_id: s ? (s.last_event_id ?? null) : null,
      ready_deadline_at: s ? (s.ready_deadline_utc ?? null) : null,
    };
  });

  const gamesOut = games.map((g) => ({
    gid: g.gid,
    seed: g.seed,
    win: g.winner_team,
    line: g.line,
    full: true,
  }));

  // Partial scores while a run is in progress, full when done; {} until any game exists.
  const scores = hasGames ? await scoreRun(runId) : {};
  const recentEventRows = (r.status === 'running' || signups.length > 0)
    ? await q(
        `SELECT * FROM run_events WHERE run_id = $1 ORDER BY seq DESC LIMIT 120`,
        [runId],
      )
    : [];
  recentEventRows.reverse();

  return send(res, 200, {
    id: r.id,
    game: r.game,
    label: r.label,
    status: r.status,
    nGames: Number(r.n_games),
    players: Number(r.players),
    seedBase: Number(r.seed_base),
    deckPreset: r.deck_preset || (r.game === 'onuw' ? DEFAULT_DECK_PRESET : null),
    deck: deckForApi(r.game, Number(r.players), r.deck_preset),
    created: r.created,
    agents,
    teamSplit,
    games: gamesOut,
    runConfig: metadata.run_config || {},
    runConfigOverrides: metadata.run_config_overrides || {},
    recentEvents: recentEventRows.map((row) => apiEvent(row)),
    connected: signups.length > 0,
    connectedSummary: {
      signups: signups.length,
      ready: signups.filter((s) => s.status === 'ready' || s.status === 'active' || s.status === 'completed').length,
      active: signups.filter((s) => s.status === 'active').length,
      completed: signups.filter((s) => s.status === 'completed').length,
    },
    scores,
  });
}
