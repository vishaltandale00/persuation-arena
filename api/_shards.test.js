// Unit tests for the pure shard-aware read helpers (api/_shards.js) and the multi-run scorecard
// aggregation (api/_read.js aggregateScorecard / scoreRuns). These mirror the Python reference in
// arena/server.py (_source_run_ids, _parent_signups, _parent_recent_events, api_runs parent branch,
// api_run parent branch, api_game cross-child resolution), arena/sharded.py (rollup_parent_status),
// and arena/store.py (list_open_runs INV-4 filter, create_signup run_kind/join_token gating).
//
// No DB: every helper takes fake row arrays / status lists, so `node --test` runs offline.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  runKind,
  sourceRunIds,
  rollupParentStatus,
  aggregateIndexRow,
  aggregateParentDetail,
  parentSignups,
  parentRecentEvents,
  isDiscoverableOpenRun,
  signupGateError,
} from './_shards.js';
import { aggregateScorecard } from './_read.js';

// --- runKind: defaults to 'normal' ----------------------------------------------------------------
test('runKind defaults missing/empty to normal', () => {
  assert.equal(runKind({}), 'normal');
  assert.equal(runKind({ run_kind: null }), 'normal');
  assert.equal(runKind({ run_kind: '' }), 'normal');
  assert.equal(runKind({ run_kind: 'parent' }), 'parent');
  assert.equal(runKind({ run_kind: 'child' }), 'child');
});

// --- sourceRunIds: a parent is backed by its children; everything else by itself -----------------
test('sourceRunIds: parent -> children, normal/child -> itself', () => {
  assert.deepEqual(
    sourceRunIds({ id: 'p', run_kind: 'parent' }, ['p_shard_0', 'p_shard_1']),
    ['p_shard_0', 'p_shard_1'],
  );
  assert.deepEqual(sourceRunIds({ id: 'r', run_kind: 'normal' }, []), ['r']);
  assert.deepEqual(sourceRunIds({ id: 'c', run_kind: 'child' }, ['ignored']), ['c']);
  // parent with no children yet -> empty source list (mirrors `store.child_run_ids() or []`)
  assert.deepEqual(sourceRunIds({ id: 'p', run_kind: 'parent' }, []), []);
});

// --- rollupParentStatus: SPEC §6.4 / REQ-2 / V-2 -------------------------------------------------
test('rollupParentStatus: no children -> open', () => {
  assert.equal(rollupParentStatus([]), 'open');
});
test('rollupParentStatus: any partial -> partial (even if some done)', () => {
  assert.equal(rollupParentStatus(['done', 'partial', 'running']), 'partial');
  assert.equal(rollupParentStatus(['partial']), 'partial');
});
test('rollupParentStatus: all done -> done', () => {
  assert.equal(rollupParentStatus(['done', 'done']), 'done');
});
test('rollupParentStatus: any not-yet-done -> running', () => {
  assert.equal(rollupParentStatus(['done', 'running']), 'running');
  assert.equal(rollupParentStatus(['open', 'open']), 'running');
  assert.equal(rollupParentStatus(['waiting', 'done']), 'running');
});
test('rollupParentStatus: missing/empty child status treated as open -> running', () => {
  assert.equal(rollupParentStatus([null, 'done']), 'running');
});

// --- aggregateIndexRow: the /api/runs parent branch ----------------------------------------------
test('aggregateIndexRow: parent rolls status up and sums children team splits', () => {
  const parent = { id: 'p', run_kind: 'parent', status: 'open' };
  const children = [
    { id: 'p_shard_0', status: 'done', team_split: { good: 3, evil: 1 } },
    { id: 'p_shard_1', status: 'partial', team_split: { good: 2, evil: 4 } },
  ];
  const { status, teamSplit } = aggregateIndexRow(parent, children);
  assert.equal(status, 'partial'); // any child partial
  assert.deepEqual(teamSplit, { good: 5, evil: 5 });
});

test('aggregateIndexRow: parent with no children -> open status, zeroed split', () => {
  const { status, teamSplit } = aggregateIndexRow({ id: 'p', run_kind: 'parent', status: 'open' }, []);
  assert.equal(status, 'open');
  assert.deepEqual(teamSplit, { good: 0, evil: 0 });
});

// --- aggregateParentDetail: the /api/runs/{id} parent branch -------------------------------------
test('aggregateParentDetail: games unioned & sorted by gid, wins/team_split summed, status rolled up', () => {
  const parent = { id: 'p', run_kind: 'parent', status: 'open' };
  const childRuns = [
    {
      id: 'p_shard_0', status: 'done',
      games: [
        { gid: 0, seed: 10, winner_team: 'good', line: 'a' },
        { gid: 2, seed: 12, winner_team: 'evil', line: 'c' },
      ],
      wins: { Alice: 2, Bob: 0 },
      team_split: { good: 1, evil: 1 },
    },
    {
      id: 'p_shard_1', status: 'running',
      games: [
        { gid: 1, seed: 11, winner_team: 'good', line: 'b' },
      ],
      wins: { Alice: 1, Bob: 1 },
      team_split: { good: 1, evil: 0 },
    },
  ];
  const agg = aggregateParentDetail(parent, childRuns);
  // union sorted by gid (disjoint global gids, D8)
  assert.deepEqual(agg.games.map((g) => g.gid), [0, 1, 2]);
  // per-name wins summed across shards
  assert.deepEqual(agg.wins, { Alice: 3, Bob: 1 });
  // team_split summed
  assert.deepEqual(agg.team_split, { good: 2, evil: 1 });
  // status rolled up (one running -> running)
  assert.equal(agg.status, 'running');
});

// --- parentSignups: one logical signup per agent_id, most-advanced status ------------------------
test('parentSignups: collapses per agent_id to the most-advanced status across shards', () => {
  // Alice is active on shard1 but only ready on shard0 -> active wins.
  // Bob is waiting on both -> waiting.
  const byChild = [
    [
      { id: 's0a', agent_id: 'Alice', status: 'ready', seat: 0 },
      { id: 's0b', agent_id: 'Bob', status: 'waiting', seat: 1 },
    ],
    [
      { id: 's1a', agent_id: 'Alice', status: 'active', seat: 0 },
      { id: 's1b', agent_id: 'Bob', status: 'waiting', seat: 1 },
    ],
  ];
  const out = parentSignups(byChild);
  const byAgent = Object.fromEntries(out.map((s) => [s.agent_id, s]));
  assert.equal(out.length, 2);
  assert.equal(byAgent.Alice.status, 'active');
  assert.equal(byAgent.Alice.id, 's1a'); // the more-advanced row is the one kept
  assert.equal(byAgent.Bob.status, 'waiting');
});

test('parentSignups: completed beats active beats ready; expired/cancelled rank lowest', () => {
  const byChild = [
    [{ id: 'a', agent_id: 'X', status: 'completed' }],
    [{ id: 'b', agent_id: 'X', status: 'active' }],
    [{ id: 'c', agent_id: 'X', status: 'expired' }],
  ];
  const out = parentSignups(byChild);
  assert.equal(out.length, 1);
  assert.equal(out[0].status, 'completed');
});

// --- parentRecentEvents: union, globally ordered, capped at 120 ----------------------------------
test('parentRecentEvents: union of children ordered by created_utc then seq, capped to last 120', () => {
  const byChild = [
    [
      { event_id: 'e1', created_utc: '2026-01-01T00:00:01.000000Z', seq: 1 },
      { event_id: 'e3', created_utc: '2026-01-01T00:00:03.000000Z', seq: 1 },
    ],
    [
      { event_id: 'e2', created_utc: '2026-01-01T00:00:02.000000Z', seq: 1 },
      // same created_utc as e3 -> tiebreak on per-shard seq
      { event_id: 'e3b', created_utc: '2026-01-01T00:00:03.000000Z', seq: 5 },
    ],
  ];
  const out = parentRecentEvents(byChild);
  assert.deepEqual(out.map((e) => e.event_id), ['e1', 'e2', 'e3', 'e3b']);
});

test('parentRecentEvents: keeps only the last 120 after global sort', () => {
  const many = [];
  for (let i = 0; i < 200; i++) {
    many.push({ event_id: `e${i}`, created_utc: `2026-01-01T00:00:00.${String(i).padStart(6, '0')}Z`, seq: i });
  }
  const out = parentRecentEvents([many]);
  assert.equal(out.length, 120);
  assert.equal(out[0].event_id, 'e80'); // 200 - 120
  assert.equal(out[119].event_id, 'e199');
});

// --- isDiscoverableOpenRun: INV-4 (only run_kind=='normal' is discoverable) -----------------------
test('isDiscoverableOpenRun: only normal runs are discoverable; parent/child excluded', () => {
  assert.equal(isDiscoverableOpenRun({ run_kind: 'normal' }), true);
  assert.equal(isDiscoverableOpenRun({ run_kind: null }), true);
  assert.equal(isDiscoverableOpenRun({}), true);
  assert.equal(isDiscoverableOpenRun({ run_kind: 'parent' }), false);
  assert.equal(isDiscoverableOpenRun({ run_kind: 'child' }), false);
});

// --- signupGateError: create_signup run_kind/join_token gating (INV-4 / SPEC D7) -----------------
test('signupGateError: parent is never joinable', () => {
  assert.equal(signupGateError({ run_kind: 'parent', join_token: 'tok' }, 'tok'), 'run_not_joinable');
  assert.equal(signupGateError({ run_kind: 'parent' }, null), 'run_not_joinable');
});
test('signupGateError: child joinable only with matching join_token', () => {
  assert.equal(signupGateError({ run_kind: 'child', join_token: 'tok' }, 'tok'), null);
  assert.equal(signupGateError({ run_kind: 'child', join_token: 'tok' }, 'wrong'), 'run_not_joinable');
  assert.equal(signupGateError({ run_kind: 'child', join_token: 'tok' }, null), 'run_not_joinable');
  // child with no expected token is never joinable
  assert.equal(signupGateError({ run_kind: 'child', join_token: null }, null), 'run_not_joinable');
});
test('signupGateError: normal runs ignore the token entirely (INV-2)', () => {
  assert.equal(signupGateError({ run_kind: 'normal' }, null), null);
  assert.equal(signupGateError({ run_kind: 'normal' }, 'whatever'), null);
  assert.equal(signupGateError({}, null), null);
});

// --- aggregateScorecard / scoreRuns: parent scorecard unions rows across shards ------------------
test('aggregateScorecard: groups by gid; disjoint gids across shards are not double-counted', () => {
  // Two shards' game_players rows unioned. gids are disjoint (D8): shard0 -> gid 0, shard1 -> gid 1.
  const rows = [
    // shard0, gid 0: Alice(good) wins, Bob(evil) loses
    { gid: 0, agent: 'Alice', team: 'good', won: 1, dealt_role: 'Seer', calls: 4, forfeits: 0 },
    { gid: 0, agent: 'Bob', team: 'evil', won: 0, dealt_role: 'Werewolf', calls: 4, forfeits: 1 },
    // shard1, gid 1: Alice(evil) wins, Bob(good) loses
    { gid: 1, agent: 'Alice', team: 'evil', won: 1, dealt_role: 'Werewolf', calls: 4, forfeits: 0 },
    { gid: 1, agent: 'Bob', team: 'good', won: 0, dealt_role: 'Villager', calls: 4, forfeits: 0 },
  ];
  const sc = aggregateScorecard(rows);
  assert.equal(sc.Alice.overall.w, 2);
  assert.equal(sc.Alice.overall.n, 2);
  assert.equal(sc.Alice.good.w, 1);
  assert.equal(sc.Alice.evil.w, 1);
  assert.equal(sc.Bob.overall.w, 0);
  assert.equal(sc.Bob.overall.n, 2);
  assert.equal(sc.Bob.forfeits, 1);
  assert.equal(sc.Bob.calls, 8);
  // forfeit_rate = 1/8
  assert.equal(sc.Bob.forfeit_rate, Math.round((1 / 8) * 1000) / 1000);
});

test('aggregateScorecard: no-contest games (no evil seat, no winner) are dropped', () => {
  const rows = [
    // gid 0: a real contested game
    { gid: 0, agent: 'A', team: 'good', won: 1, dealt_role: 'Seer', calls: 1, forfeits: 0 },
    { gid: 0, agent: 'B', team: 'evil', won: 0, dealt_role: 'Werewolf', calls: 1, forfeits: 0 },
    // gid 1: no evil seat AND no winner -> dropped from scoring
    { gid: 1, agent: 'A', team: 'good', won: 0, dealt_role: 'Villager', calls: 1, forfeits: 0 },
    { gid: 1, agent: 'B', team: 'good', won: 0, dealt_role: 'Villager', calls: 1, forfeits: 0 },
  ];
  const sc = aggregateScorecard(rows);
  // only the contested game counts toward n
  assert.equal(sc.A.overall.n, 1);
  assert.equal(sc.B.overall.n, 1);
});
