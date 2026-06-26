import { strToU8, zipSync } from 'fflate';

export const REPLAY_SCHEMA_VERSION = 'pa-replay-v0.1';
export const PERSPECTIVE_SCHEMA_VERSION = 'pa-replay-perspective-v0.1';

const PUBLIC_GAME_EVENT_TYPES = new Set([
  'game_setup',
  'phase_started',
  'phase_ended',
  'speech',
  'pass',
  'discussion_notice',
  'vote_revealed',
  'game_result',
]);

const REDACTED_PUBLIC_PAYLOAD_KEYS = new Set([
  'reasoning',
  'declared_reasoning',
  'provider_reasoning',
  'provider_reasoning_details',
  'raw',
  'raw_model_output',
  'prompt',
  'observation',
  'legal_action',
]);

export function gameInstanceId(runId, gid) {
  return `${runId}_game_${String(gid).padStart(3, '0')}`;
}

export function gameFilePath(runId, gid) {
  return `games/${runId}__g${String(gid).padStart(3, '0')}.json`;
}

export function perspectiveFilePath(runId, gid, seat) {
  return `perspectives/${runId}__g${String(gid).padStart(3, '0')}__seat${seat}.json`;
}

function parseJson(value, fallback = null) {
  if (value == null || value === '') return fallback;
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function asInt(value, fallback = 0) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : fallback;
}

function asBool(value) {
  if (value === true || value === false) return value;
  if (value == null) return false;
  return Number(value) === 1 || String(value).toLowerCase() === 'true';
}

function sortedBySeq(rows) {
  return [...rows].sort((a, b) => asInt(a.seq) - asInt(b.seq) || String(a.id || '').localeCompare(String(b.id || '')));
}

function sortedBySeat(rows) {
  return [...rows].sort((a, b) => asInt(a.seat) - asInt(b.seat));
}

function sortedByCreated(rows) {
  return [...rows].sort((a, b) => {
    const ac = String(a.created_utc || '');
    const bc = String(b.created_utc || '');
    if (ac !== bc) return ac < bc ? -1 : 1;
    return String(a.id || a.turn_id || '').localeCompare(String(b.id || b.turn_id || ''));
  });
}

function rowsBy(rows, keyFn) {
  const out = new Map();
  for (const row of rows || []) {
    const key = keyFn(row);
    if (key == null) continue;
    if (!out.has(key)) out.set(key, []);
    out.get(key).push(row);
  }
  return out;
}

function rowBy(rows, keyFn) {
  const out = new Map();
  for (const row of rows || []) {
    const key = keyFn(row);
    if (key != null && !out.has(key)) out.set(key, row);
  }
  return out;
}

function redactPublicPayload(value) {
  if (Array.isArray(value)) return value.map(redactPublicPayload);
  if (!value || typeof value !== 'object') return value;
  const out = {};
  for (const [key, child] of Object.entries(value)) {
    if (REDACTED_PUBLIC_PAYLOAD_KEYS.has(key)) continue;
    out[key] = redactPublicPayload(child);
  }
  return out;
}

function eventPayload(event) {
  return parseJson(event.payload_json ?? event.payload, {});
}

function normalizePublicEvent(event) {
  const payload = redactPublicPayload(eventPayload(event));
  const out = {
    event_id: event.id ?? event.event_id ?? null,
    seq: asInt(event.seq),
    phase: event.phase ?? null,
    type: event.type,
  };
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    if (payload.actor_seat != null) out.seat = asInt(payload.actor_seat);
    if (event.type === 'speech' && payload.text != null) {
      out.action = { speak: String(payload.text) };
      const rest = { ...payload };
      delete rest.actor_seat;
      delete rest.text;
      if (Object.keys(rest).length) out.payload = rest;
      return out;
    }
    if (event.type === 'pass' && payload.actor_seat != null) {
      out.action = { pass: true };
    }
  }
  out.payload = payload;
  return out;
}

function normalizePrivateEvent(event) {
  return {
    event_id: event.id ?? event.event_id ?? null,
    seq: asInt(event.seq),
    phase: event.phase ?? null,
    type: event.type,
    payload: eventPayload(event),
  };
}

function publicEventsFromTranscript(transcript) {
  const out = [];
  let seq = 1;
  for (const phase of transcript?.phases || []) {
    const phaseKind = String(phase.kind || phase.name || '').toLowerCase();
    if (!['discussion', 'talk', 'vote', 'result'].includes(phaseKind)) continue;
    const phaseName = phaseKind === 'talk' ? 'discussion' : phaseKind;
    for (const event of phase.events || []) {
      if (!event || typeof event !== 'object') continue;
      const base = {
        event_id: null,
        seq: seq++,
        phase: phaseName,
        source: 'transcript_json',
      };
      if (event.t === 'say' && event.pid != null) {
        out.push({
          ...base,
          type: 'speech',
          seat: asInt(event.pid),
          action: { speak: String(event.text || '') },
          payload: event.urgency == null ? {} : { urgency: event.urgency },
        });
      } else if (event.t === 'pass' && event.pid != null) {
        out.push({
          ...base,
          type: 'pass',
          seat: asInt(event.pid),
          action: { pass: true },
          payload: event.stance == null ? {} : { stance: event.stance },
        });
      } else if (event.t === 'vote' && event.pid != null) {
        out.push({
          ...base,
          type: 'vote_revealed',
          seat: asInt(event.pid),
          payload: { target: event.tgt ?? null },
        });
      } else if (event.t === 'sys' && event.text) {
        out.push({
          ...base,
          type: 'discussion_notice',
          payload: { text: String(event.text) },
        });
      }
    }
  }
  return out;
}

function eventTargetsPerspective(event, target) {
  if (event.visibility !== 'private') return false;
  if (event.target_signup_id != null && target.signup_id != null) {
    return String(event.target_signup_id) === String(target.signup_id);
  }
  if (event.target_signup_id != null) return false;
  const payload = eventPayload(event);
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return false;
  const seat = payload.target_seat ?? payload.seat;
  return seat != null && Number(seat) === Number(target.seat);
}

function participantFromGamePlayer(player, signupById, signupBySeat, agentById) {
  const signup = player.signup_id ? signupById.get(player.signup_id) : signupBySeat.get(asInt(player.seat));
  const agent = player.agent_id ? agentById.get(player.agent_id) : agentById.get(signup?.agent_id);
  return {
    seat: asInt(player.seat),
    public_name: player.agent ?? agent?.display_name ?? signup?.display_name ?? `Seat ${asInt(player.seat)}`,
    agent_id: player.agent_id ?? signup?.agent_id ?? null,
    signup_id: player.signup_id ?? signup?.id ?? null,
    model: player.model ?? agent?.declared_model ?? signup?.declared_model ?? null,
  };
}

function participantFromSignup(signup, agentById) {
  const agent = agentById.get(signup.agent_id);
  return {
    seat: asInt(signup.seat),
    public_name: signup.display_name ?? agent?.display_name ?? `Seat ${asInt(signup.seat)}`,
    agent_id: signup.agent_id ?? agent?.id ?? null,
    signup_id: signup.id ?? null,
    model: agent?.declared_model ?? signup.declared_model ?? null,
  };
}

function participantFromTranscript(player, index) {
  return {
    seat: asInt(player.seat ?? player.pid ?? index),
    public_name: player.name ?? player.agent ?? `Seat ${asInt(player.seat ?? player.pid ?? index)}`,
    agent_id: player.agent_id ?? null,
    signup_id: player.signup_id ?? null,
    model: player.model ?? null,
  };
}

function playerLabels(players) {
  return sortedBySeat(players).map((p) => ({
    seat: asInt(p.seat),
    dealt_role: p.dealt_role ?? p.dealt ?? null,
    end_role: p.end_role ?? p.end ?? null,
    team: p.team ?? null,
    won: asBool(p.won),
    calls: asInt(p.calls),
    forfeits: asInt(p.forfeits),
    agent_id: p.agent_id ?? null,
    signup_id: p.signup_id ?? null,
  }));
}

function labelForSeat(players, seat, winnerTeam) {
  const row = players.find((p) => Number(p.seat) === Number(seat));
  if (!row) {
    return {
      dealt_role: null,
      end_role: null,
      team: null,
      won: false,
      winner_team: winnerTeam ?? null,
      calls: 0,
      forfeits: 0,
      forfeit_rate: 0,
    };
  }
  const calls = asInt(row.calls);
  const forfeits = asInt(row.forfeits);
  return {
    dealt_role: row.dealt_role ?? row.dealt ?? null,
    end_role: row.end_role ?? row.end ?? null,
    team: row.team ?? null,
    won: asBool(row.won),
    winner_team: winnerTeam ?? null,
    calls,
    forfeits,
    forfeit_rate: calls ? Number((forfeits / calls).toFixed(3)) : 0,
  };
}

function setupFromEvents(publicEvents) {
  const setup = publicEvents.find((e) => e.type === 'game_setup');
  const payload = setup?.payload ?? {};
  return {
    deck: payload.deck ?? null,
    rules_payload: payload.rules_payload ?? {},
    win_condition: payload.win_condition ?? null,
    player_count: payload.n ?? null,
    center_count: payload.center_count ?? null,
  };
}

function outcomeFrom(gameRow, transcript) {
  const outcome = transcript?.outcome && typeof transcript.outcome === 'object'
    ? transcript.outcome
    : {};
  return {
    winner_team: gameRow.winner_team ?? transcript?.winner_team ?? outcome.winner_team ?? null,
    text: gameRow.line ?? outcome.text ?? transcript?.line ?? null,
    deaths: transcript?.deaths ?? outcome.deaths ?? [],
    votes: transcript?.votes ?? outcome.votes ?? [],
  };
}

function buildParticipants({ gamePlayers, transcript, signupsForRun, agentById }) {
  if (gamePlayers.length) {
    const signupById = rowBy(signupsForRun, (s) => s.id);
    const signupBySeat = rowBy(signupsForRun.filter((s) => s.seat != null), (s) => asInt(s.seat));
    return sortedBySeat(gamePlayers).map((p) => participantFromGamePlayer(p, signupById, signupBySeat, agentById));
  }

  const seatedSignups = signupsForRun.filter((s) => s.seat != null);
  if (seatedSignups.length) {
    return sortedBySeat(seatedSignups).map((s) => participantFromSignup(s, agentById));
  }

  const transcriptPlayers = transcript?.players || [];
  return transcriptPlayers.map((p, i) => participantFromTranscript(p, i));
}

function buildDecision({ turn, reply, allEventsForGame }) {
  const action = parseJson(reply?.action_json ?? reply?.action, null);
  const observation = parseJson(turn.observation_json ?? turn.observation, null);
  const legalAction = parseJson(turn.legal_action_json ?? turn.legal_action, null);
  const turnEvent = allEventsForGame.find((event) => {
    if (event.type !== 'private_observation') return false;
    const payload = eventPayload(event);
    return payload?.turn_id === turn.id;
  });
  const cursorSeq = turnEvent ? asInt(turnEvent.seq) : null;
  const accepted = reply ? asBool(reply.accepted) : false;
  const defaulted = turn.status === 'defaulted' || (reply != null && !accepted);
  return {
    turn_id: turn.id,
    phase: turn.phase ?? null,
    action_kind: turn.action_kind ?? null,
    event_context: {
      public_events_before_seq: cursorSeq,
      private_events_before_seq: cursorSeq,
      cursor_semantics: 'exclusive_seq',
      missing_turn_event: cursorSeq == null,
    },
    observation,
    legal_action: legalAction,
    output: {
      declared_reasoning: reply?.reasoning ?? null,
      action,
      accepted,
      defaulted,
      status: turn.status ?? null,
      client_ms: reply?.client_ms ?? null,
      reply_created_utc: reply?.created_utc ?? null,
    },
  };
}

export function buildReplayDataset(inputRows, options = {}) {
  const rows = {
    runs: inputRows.runs || [],
    games: inputRows.games || [],
    gamePlayers: inputRows.gamePlayers || inputRows.game_players || [],
    runEvents: inputRows.runEvents || inputRows.run_events || [],
    turns: inputRows.turns || [],
    turnReplies: inputRows.turnReplies || inputRows.turn_replies || [],
    agents: inputRows.agents || [],
    runSignups: inputRows.runSignups || inputRows.run_signups || [],
  };

  const createdUtc = options.createdUtc || new Date().toISOString();
  const requestedRunIds = options.runIds || null;
  const files = {};
  const gameFiles = [];
  const perspectiveFiles = [];
  const skippedGames = [];
  const skippedPerspectives = [];

  const gamesByRun = rowsBy(rows.games, (g) => g.run_id);
  const playersByRunGid = rowsBy(rows.gamePlayers, (p) => `${p.run_id}:${p.gid}`);
  const eventsByGameInstance = rowsBy(rows.runEvents, (e) => e.game_instance_id);
  const turnsByGameInstance = rowsBy(rows.turns, (t) => t.game_instance_id);
  const repliesByTurnId = rowBy(rows.turnReplies, (r) => r.turn_id);
  const signupsByRun = rowsBy(rows.runSignups, (s) => s.run_id);
  const agentById = rowBy(rows.agents, (a) => a.id);

  const includedRunIds = new Set();

  for (const run of [...rows.runs].sort((a, b) => String(a.id).localeCompare(String(b.id)))) {
    const runGames = [...(gamesByRun.get(run.id) || [])].sort((a, b) => asInt(a.gid) - asInt(b.gid));
    if (!runGames.length) continue;
    includedRunIds.add(run.id);
    const signupsForRun = signupsByRun.get(run.id) || [];

    for (const gameRow of runGames) {
      const gid = asInt(gameRow.gid);
      const iid = gameInstanceId(run.id, gid);
      const transcript = parseJson(gameRow.transcript_json ?? gameRow.transcript, {});
      const gamePlayers = playersByRunGid.get(`${run.id}:${gid}`) || [];
      const allEventsForGame = sortedBySeq(eventsByGameInstance.get(iid) || []);
      let publicEvents = allEventsForGame
        .filter((event) => event.visibility === 'public' && PUBLIC_GAME_EVENT_TYPES.has(event.type))
        .map(normalizePublicEvent);
      if (!publicEvents.length) publicEvents = publicEventsFromTranscript(transcript);
      const participants = buildParticipants({ gamePlayers, transcript, signupsForRun, agentById });
      const outcome = outcomeFrom(gameRow, transcript);
      const gamePath = gameFilePath(run.id, gid);
      const traceQuality = allEventsForGame.length ? 'event_stream' : 'transcript_only';

      const gameDoc = {
        schema_version: REPLAY_SCHEMA_VERSION,
        game_id: `${run.id}:g${String(gid).padStart(3, '0')}`,
        run_id: run.id,
        gid,
        game_instance_id: iid,
        game: run.game ?? null,
        seed: gameRow.seed ?? transcript?.seed ?? null,
        deck_preset: run.deck_preset ?? null,
        trace_quality: traceQuality,
        participants,
        setup: setupFromEvents(publicEvents),
        public_events: publicEvents,
        outcome,
        player_labels: playerLabels(gamePlayers),
        provenance: {
          source_tables: ['runs', 'games', 'game_players', 'run_events'],
        },
      };
      files[gamePath] = gameDoc;
      gameFiles.push(gamePath);

      const turnsForGame = sortedByCreated(turnsByGameInstance.get(iid) || []);
      if (!turnsForGame.length) {
        skippedGames.push({
          run_id: run.id,
          gid,
          game_ref: gamePath,
          reason: 'missing_turn_trace',
          trace_quality: traceQuality,
        });
        continue;
      }

      for (const target of participants) {
        const targetTurns = turnsForGame.filter((turn) => {
          if (Number(turn.seat) !== Number(target.seat)) return false;
          if (target.signup_id && turn.signup_id && String(turn.signup_id) !== String(target.signup_id)) return false;
          return true;
        });
        if (!targetTurns.length) {
          skippedPerspectives.push({
            run_id: run.id,
            gid,
            seat: target.seat,
            game_ref: gamePath,
            reason: 'missing_target_decisions',
          });
          continue;
        }
        const privateEvents = allEventsForGame
          .filter((event) => eventTargetsPerspective(event, target))
          .map(normalizePrivateEvent);
        const decisions = targetTurns.map((turn) => buildDecision({
          turn,
          reply: repliesByTurnId.get(turn.id),
          allEventsForGame,
        }));
        const perspectivePath = perspectiveFilePath(run.id, gid, target.seat);
        files[perspectivePath] = {
          trace_id: `${run.id}:g${String(gid).padStart(3, '0')}:seat${target.seat}`,
          purpose: 'agent_improvement_replay',
          schema_version: PERSPECTIVE_SCHEMA_VERSION,
          game_ref: `../${gamePath}`,
          target,
          private_events: privateEvents,
          target_decisions: decisions,
          labels: labelForSeat(gamePlayers, target.seat, outcome.winner_team),
          provenance: {
            source_tables: ['turns', 'turn_replies', 'run_events', 'game_players'],
          },
        };
        perspectiveFiles.push(perspectivePath);
      }
    }
  }

  const maxRunCreatedUtc = rows.runs
    .map((r) => r.created_utc || r.created)
    .filter(Boolean)
    .sort()
    .at(-1) || null;

  const manifest = {
    schema_version: REPLAY_SCHEMA_VERSION,
    created_utc: createdUtc,
    source: {
      kind: options.sourceKind || 'neon',
      database: options.sourceDatabase || 'DATABASE_URL environment',
      max_run_created_utc: maxRunCreatedUtc,
    },
    filters: {
      statuses: options.statuses || ['done'],
      run_ids: requestedRunIds,
      games: 'complete_with_trace',
    },
    cursor_semantics: {
      public_events_before_seq: 'exclusive: replay public_events with seq < cursor',
      private_events_before_seq: 'exclusive: replay private_events with seq < cursor',
    },
    counts: {
      runs: includedRunIds.size,
      games: gameFiles.length,
      perspectives: perspectiveFiles.length,
      skipped_games: skippedGames.length,
      skipped_perspectives: skippedPerspectives.length,
    },
    files: {
      games: gameFiles,
      perspectives: perspectiveFiles,
    },
    skips: skippedGames,
    skipped_perspectives: skippedPerspectives,
  };
  files['manifest.json'] = manifest;

  return {
    manifest,
    files,
    games: gameFiles.map((path) => files[path]),
    perspectives: perspectiveFiles.map((path) => files[path]),
  };
}

export function zipReplayDataset(files) {
  const zippable = {};
  for (const [path, value] of Object.entries(files)) {
    const body = `${JSON.stringify(value, null, 2)}\n`;
    zippable[path] = strToU8(body);
  }
  return Buffer.from(zipSync(zippable, { level: 6 }));
}
