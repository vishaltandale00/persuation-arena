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
import { scoreRun, scoreRuns, deckForApi, DEFAULT_DECK_PRESET, apiEvent } from '../../_read.js';
import {
  runKind, sourceRunIds, aggregateParentDetail, parentSignups, parentRecentEvents,
} from '../../_shards.js';

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

  // SPEC D7: a sharded PARENT presents as ONE normal run, sourcing every aggregate field from its
  // children (the parent row owns no games/signups/events/wins of its own). source_ids = the child
  // run ids in shard order for a parent, else [runId] for a normal/child run.
  const isParent = runKind(r) === 'parent';
  let childIds = [];
  if (isParent) {
    childIds = (await q('SELECT id FROM runs WHERE parent_run_id = $1 ORDER BY shard_index', [runId]))
      .map((c) => c.id);
  }
  const sourceIds = sourceRunIds(r, childIds);

  // store.get_run, per source run: games (ordered by gid), per-agent wins, and the winner-team split.
  async function loadRunDetail(srcId, baseAgents) {
    const detailGames = await q(
      'SELECT gid, seed, winner_team, line FROM games WHERE run_id = $1 ORDER BY gid',
      [srcId],
    );
    const detailWins = {};
    for (const a of baseAgents) detailWins[a.name] = 0;
    const wr = await q(
      'SELECT agent, SUM(won) w FROM game_players WHERE run_id = $1 GROUP BY agent',
      [srcId],
    );
    for (const row of wr) detailWins[row.agent] = Number(row.w) || 0;
    const detailSplit = { good: 0, evil: 0 };
    for (const g of detailGames) detailSplit[g.winner_team] = (detailSplit[g.winner_team] || 0) + 1;
    return { id: srcId, games: detailGames, wins: detailWins, team_split: detailSplit };
  }

  let games;
  let wins;
  let teamSplit;
  let status = r.status;
  if (isParent) {
    // children carry the same agents_json as the parent (REQ-5), so seed each child's wins map from
    // the parent roster; aggregateParentDetail unions games (disjoint global gids, D8) and sums
    // wins/team_split, and rolls the status up from the children.
    const childRuns = [];
    for (const cid of sourceIds) {
      const detail = await loadRunDetail(cid, agentsSrc);
      // aggregateParentDetail rolls the parent's status up from each child's `status`, but
      // loadRunDetail only reads games/wins/team_split — it never loads the child run's status. Load
      // it here (the child's own runs row) so the rollup sees the real per-shard status instead of
      // undefined (which rollupParentStatus treats as 'open' -> parent stuck at 'running'). This makes
      // the detail path agree with the run-list index path (childRunsForIndex selects status).
      const childRow = (await q('SELECT status FROM runs WHERE id = $1', [cid]))[0];
      detail.status = childRow ? childRow.status : null;
      childRuns.push(detail);
    }
    const agg = aggregateParentDetail(r, childRuns);
    games = agg.games;
    wins = agg.wins;
    teamSplit = agg.team_split;
    status = agg.status;
    // a parent's roster needs every name to default to 0 wins even if no shard recorded it yet.
    for (const a of agentsSrc) if (!(a.name in wins)) wins[a.name] = 0;
  } else {
    const detail = await loadRunDetail(runId, agentsSrc);
    games = detail.games;
    wins = detail.wins;
    teamSplit = detail.team_split;
  }
  const hasGames = games.length > 0;

  // store.list_run_signups: signups joined to agents, ordered seat-first. For a parent the children
  // share identities (REQ-5), so load each child's signups and collapse to one logical signup per
  // agent_id (most-advanced status across shards) via parentSignups (server.py _parent_signups).
  async function loadSignups(srcId) {
    const rows = await q(
      `SELECT s.*, a.display_name, a.status AS agent_status, a.last_seen_utc
         FROM run_signups s JOIN agents a ON a.id = s.agent_id
        WHERE s.run_id = $1
        ORDER BY s.seat IS NULL, s.seat, s.created_utc`,
      [srcId],
    );
    // store.list_run_signups runs _expire_signup_if_needed on every signup, lazily flipping
    // deadline-expired 'waiting'/'ready_required' rows to 'expired' (and persisting), so every
    // downstream read of s.status sees the effective status.
    for (const s of rows) await expireSignupIfNeeded(s);
    return rows;
  }

  let signups;
  if (isParent) {
    const signupsByChild = [];
    for (const cid of sourceIds) signupsByChild.push(await loadSignups(cid));
    signups = parentSignups(signupsByChild);
  } else {
    signups = await loadSignups(runId);
  }
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

  // Partial scores while a run is in progress, full when done; {} until any game exists. A parent
  // aggregates across its shards (scoreRuns unions game_players rows; disjoint global gids, D8).
  const scores = hasGames ? (isParent ? await scoreRuns(sourceIds) : await scoreRun(runId)) : {};

  // Recent events: a normal/child run streams its own last 120; a parent unions its children's
  // recent events, globally ordered (created_utc, then per-shard seq) and capped to the last 120
  // (server.py _parent_recent_events). Gate on the rolled-up status (running) or any signup.
  let recentEventRows = [];
  if (status === 'running' || signups.length > 0) {
    if (isParent) {
      const eventsByChild = [];
      for (const cid of sourceIds) {
        const rows = await q(
          'SELECT * FROM run_events WHERE run_id = $1 ORDER BY seq DESC LIMIT 120',
          [cid],
        );
        rows.reverse();
        eventsByChild.push(rows);
      }
      recentEventRows = parentRecentEvents(eventsByChild);
    } else {
      recentEventRows = await q(
        'SELECT * FROM run_events WHERE run_id = $1 ORDER BY seq DESC LIMIT 120',
        [runId],
      );
      recentEventRows.reverse();
    }
  }

  return send(res, 200, {
    id: r.id,
    game: r.game,
    label: r.label,
    status,
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
