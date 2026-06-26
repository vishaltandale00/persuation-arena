// INTEGRATION test for the GET /api/runs/{run_id} handler's PARENT status rollup (codex round-3 P1).
//
// Bug: aggregateParentDetail() rolls the parent status up from childRuns.map(c => c.status), but
// loadRunDetail() in the handler never loaded each child run's status, so every child status was
// undefined -> treated as 'open' -> the parent always reported 'running', even when its children were
// all 'done' or one was 'partial'. The list path (/api/runs) loads child status and disagreed.
//
// This drives the real handler (not just the pure helper) with a mocked DB query fn (q from _db.js,
// now lazy/exported), asserting the response's rolled-up status is correct.
//
// No real DB: mock.module replaces api/_db.js's q with a SQL-routing fake. _read.js's scoreRuns also
// imports q from _db.js, so mocking the module covers it too.
// Run with:  node --experimental-test-module-mocks --test 'api/runs/[[]run_id[]]/index.test.js'
// (mock.module is experimental; the brackets in the path are escaped for node's --test glob.) When the
// flag is absent we skip rather than fail, so a plain `node --test` over the suite stays green.
import { test, mock } from 'node:test';
import assert from 'node:assert/strict';

const DB_PATH = new URL('../../_db.js', import.meta.url).pathname;

// mock.module only exists under --experimental-test-module-mocks. Detect it so a flagless run skips.
const MODULE_MOCKS = typeof mock.module === 'function';

// Build a fake req/res pair and capture the handler's send() output.
function fakeReqRes(runId) {
  const captured = {};
  const req = { method: 'GET', query: { run_id: runId }, headers: {} };
  const res = {
    statusCode: 200,
    setHeader() {},
    end(body) { captured.body = JSON.parse(body); captured.status = this.statusCode; },
  };
  return { req, res, captured };
}

// A SQL-routing mock for q(text, params). `childStatuses` maps a child run id -> its status row value.
function makeQ(childStatuses) {
  const parentId = 'p';
  const childIds = Object.keys(childStatuses);
  return async (text, params = []) => {
    const t = text.replace(/\s+/g, ' ').trim();
    // load the run row (parent)
    if (t.startsWith('SELECT * FROM runs WHERE id =')) {
      const id = params[0];
      if (id === parentId) {
        return [{
          id: parentId, run_kind: 'parent', status: 'open', game: 'onuw',
          label: 'L', n_games: 4, players: 5, seed_base: 1, created: '2026-01-01 00:00',
          agents_json: JSON.stringify([{ name: 'Alice' }, { name: 'Bob' }]),
          metadata_json: null, deck_preset: null,
        }];
      }
      // a child run row (used by loadRunDetail's status load, once fixed)
      return [{ id, run_kind: 'child', status: childStatuses[id] }];
    }
    // child id discovery for the parent
    if (t.startsWith('SELECT id FROM runs WHERE parent_run_id =')) {
      return childIds.map((id, i) => ({ id, shard_index: i }));
    }
    // child status load (the FIX adds this; tolerate either column projection)
    if (t.includes('FROM runs WHERE id =') && t.includes('status')) {
      const id = params[0];
      return [{ id, status: childStatuses[id] }];
    }
    // per-child games (none -> keeps the test focused on status rollup)
    if (t.startsWith('SELECT gid, seed, winner_team, line FROM games')) return [];
    // per-child wins
    if (t.startsWith('SELECT agent, SUM(won)')) return [];
    // signups
    if (t.includes('FROM run_signups s JOIN agents a')) return [];
    // recent events
    if (t.startsWith('SELECT * FROM run_events')) return [];
    // game_players scoring (scoreRuns)
    if (t.includes('FROM game_players WHERE run_id =')) return [];
    return [];
  };
}

async function runHandler(childStatuses) {
  mock.module(DB_PATH, {
    namedExports: {
      q: makeQ(childStatuses),
      send(res, status, body) { res.statusCode = status; res.end(JSON.stringify(body)); },
      utcnow: () => '2026-01-01T00:00:00.000000Z',
      utcAfter: () => '2026-01-01T00:01:00.000000Z',
    },
  });
  const mod = await import(`../../runs/[run_id]/index.js?case=${encodeURIComponent(JSON.stringify(childStatuses))}`);
  const { req, res, captured } = fakeReqRes('p');
  await mod.default(req, res);
  mock.reset();
  return captured;
}

test('GET /api/runs/{parent}: all children done -> status done (not running)', { skip: !MODULE_MOCKS && 'needs --experimental-test-module-mocks' }, async () => {
  const captured = await runHandler({ p_shard_0: 'done', p_shard_1: 'done' });
  assert.equal(captured.status, 200);
  assert.equal(captured.body.status, 'done');
});

test('GET /api/runs/{parent}: one child partial -> status partial (not running)', { skip: !MODULE_MOCKS && 'needs --experimental-test-module-mocks' }, async () => {
  const captured = await runHandler({ p_shard_0: 'done', p_shard_1: 'partial' });
  assert.equal(captured.status, 200);
  assert.equal(captured.body.status, 'partial');
});

test('GET /api/runs/{parent}: a child still running -> status running', { skip: !MODULE_MOCKS && 'needs --experimental-test-module-mocks' }, async () => {
  const captured = await runHandler({ p_shard_0: 'done', p_shard_1: 'running' });
  assert.equal(captured.status, 200);
  assert.equal(captured.body.status, 'running');
});
