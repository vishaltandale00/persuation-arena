// Standalone offline assertions for api/runs/import.js's pure helpers.
//
// Runs with plain `node` — NO database. import.js imports api/_db.js, which calls
// neon(process.env.DATABASE_URL) at module load and THROWS when DATABASE_URL is unset
// (@neondatabase/serverless ^1.1.0). So we set a DUMMY DATABASE_URL here before importing; neon()
// only opens a real connection on an actual query, and we never run one — the import.js helpers
// (gameToSqlParams, runHeaderToSqlParams) are pure, and the agentFromToken auth check below only
// exercises the no-bearer path, which returns before any query.
//
//   DATABASE_URL='postgres://u:p@localhost/db' node tests/import_endpoint.test.mjs
//
// Exits non-zero on the first failed assertion (node:assert throws).
process.env.DATABASE_URL ||= 'postgres://u:p@localhost/db';

import assert from 'node:assert/strict';
import {
  gameToSqlParams, runHeaderToSqlParams,
  RUNS_UPSERT, GAMES_INSERT, PLAYERS_INSERT,
} from '../api/runs/import.js';
import { agentFromToken } from '../api/_db.js';

let passed = 0;
function ok(name, fn) {
  fn();
  passed += 1;
  console.log(`  ok - ${name}`);
}

// --- auth: agentFromToken (the registration pa_live_ token) FAILS CLOSED on no/non-bearer ------
// Only the no-bearer path is offline-safe: bearer(req)===null short-circuits before any DB query.
// A present bearer would trigger a real Neon lookup against agents.token_hash, so we don't exercise
// that path here (the handler rejects an unknown token with 403 and a valid one authorizes).
assert.equal(await agentFromToken({ headers: {} }), null);
assert.equal(await agentFromToken({ headers: { authorization: '' } }), null);
assert.equal(await agentFromToken({ headers: { authorization: 'Basic xyz' } }), null);
passed += 1;
console.log('  ok - agentFromToken returns null without a bearer (fail-closed, no DB hit)');

// --- payload -> SQL params: game_players DERIVED from transcript.players joined to agents[] ---
ok('gameToSqlParams derives game_players from transcript.players (won->0/1, int-default-0)', () => {
  const game = {
    gid: 3,
    transcript: {
      seed: 42,
      winner_team: 'good',
      outcome: { text: 'village wins' },
      players: [
        { seat: 0, dealt: 'Seer', end: 'Seer', team: 'good', won: true, calls: 2 },
        { seat: 1, dealt: 'Werewolf', end: 'Werewolf', team: 'evil', won: false },
      ],
    },
    agents: [
      { name: 'alice', model: 'm1', agent_id: 'ag0', signup_id: 'su0' },
      { name: 'bob', model: 'm2' },
    ],
  };
  const { gameParams, playerParams } = gameToSqlParams('r1', game);

  // games row: [run_id, gid, seed, winner_team, line(outcome.text), transcript_json]
  assert.equal(gameParams[0], 'r1');
  assert.equal(gameParams[1], 3);
  assert.equal(gameParams[2], 42);
  assert.equal(gameParams[3], 'good');
  assert.equal(gameParams[4], 'village wins');
  assert.equal(JSON.parse(gameParams[5]).seed, 42);

  // seat 0 joined to agents[0]; won true -> 1; calls 2; forfeits default 0; agent_id/signup_id flow.
  assert.deepEqual(playerParams[0],
    ['r1', 3, 0, 'alice', 'm1', 'Seer', 'Seer', 'good', 1, 2, 0, 'ag0', 'su0']);
  // seat 1 joined to agents[1]; won false -> 0; missing calls -> 0; agent_id/signup_id -> null.
  assert.deepEqual(playerParams[1],
    ['r1', 3, 1, 'bob', 'm2', 'Werewolf', 'Werewolf', 'evil', 0, 0, 0, null, null]);
});

ok('gameToSqlParams throws when a seat has no matching agent', () => {
  const game = {
    gid: 1,
    transcript: { seed: 1, winner_team: 'good', outcome: { text: 't' },
      players: [{ seat: 5, dealt: 'V', end: 'V', team: 'good', won: false }] },
    agents: [{ name: 'a', model: 'm' }],  // no index 5
  };
  assert.throws(() => gameToSqlParams('r1', game), /seat 5 has no agent/);
});

// --- run header -> upsert params + SQL shape sanity ------------------------------------------
ok('runHeaderToSqlParams parameterizes status (not hardcoded queued) + TERMINAL_STATUSES tail', () => {
  const run = { id: 'r1', game: 'onuw', status: 'done', n_games: 5, players: 5,
    seed_base: 9000, agents: [{ name: 'a' }] };
  const p = runHeaderToSqlParams(run, 'owner', '2026-06-24T00:00:00Z');
  assert.equal(p[0], 'r1');
  assert.equal(p[3], 'done');                 // status slot carries the uploaded status
  assert.equal(JSON.parse(p[8])[0].name, 'a'); // agents_json
  assert.ok(Array.isArray(p[12]));             // TERMINAL_RUN_STATUSES tail for the monotonic CASE
});

ok('SQL constants carry the idempotency clauses', () => {
  assert.ok(GAMES_INSERT.includes('ON CONFLICT (run_id,gid) DO NOTHING'));
  assert.ok(GAMES_INSERT.includes('RETURNING gid'));
  assert.ok(PLAYERS_INSERT.includes('ON CONFLICT (run_id,gid,seat) DO NOTHING'));
  assert.ok(RUNS_UPSERT.includes('runs.status = ANY($13::text[])'));  // monotonic status CASE
});

console.log(`\nimport_endpoint.test.mjs: ${passed} assertions passed`);
