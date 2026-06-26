// Parity test: api/leaderboard.js assembleLeaderboard() vs the Python arena.rating.leaderboard()
// golden snapshot. computeRatings() (the Elo *compute*) is covered by rating_parity.test.mjs; this
// covers the PUBLIC PROJECTION the observer renders — _public scalars + per-objective/per-role cells
// (vs_spread, hard, conservative, banker's rounding) — which had no cross-language coverage.
//
// Feeds the fixture's recompute OUTPUTS (ratings/events/role_difficulty) into the JS projection and
// asserts it reproduces outputs.leaderboard_projection cell-for-cell.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// assembleLeaderboard is pure, but importing leaderboard.js -> _db.js needs a non-empty URL.
process.env.DATABASE_URL ||= 'postgres://u:p@localhost/db';
const { assembleLeaderboard } = await import('../api/leaderboard.js');

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'rating_golden.json'), 'utf8'));
const { leaderboard_rows, all_rating_events, role_difficulty_map, leaderboard_projection } = fixture.outputs;

const EPS = 1e-9;
function close(a, b, path) {
  if (typeof a === 'number' && typeof b === 'number') {
    assert.ok(Math.abs(a - b) <= EPS, `${path}: ${a} != ${b} (|Δ|=${Math.abs(a - b)})`);
    return;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    assert.ok(Array.isArray(a) && Array.isArray(b) && a.length === b.length,
      `${path}: array shape ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
    a.forEach((x, i) => close(x, b[i], `${path}[${i}]`));
    return;
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), `${path}: key set differs`);
    for (const k of Object.keys(a)) close(a[k], b[k], `${path}.${k}`);
    return;
  }
  assert.strictEqual(a, b, `${path}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
}

const difficulties = role_difficulty_map.filter((d) => d.bucket === '*');
const got = assembleLeaderboard(leaderboard_rows, all_rating_events, difficulties);

// 1. Content parity, keyed by identity_key. assembleLeaderboard projects each competitor
//    independently of input order; the list's RANK order comes from the SQL ORDER BY, which is
//    byte-identical to Python's store.leaderboard_rows ("ORDER BY (elo - 2*173.0*rd) DESC, games DESC").
const byKey = (xs) => Object.fromEntries(xs.map((x) => [x.identity_key, x]));
const G = byKey(got);
const E = byKey(leaderboard_projection);
assert.deepEqual(Object.keys(G).sort(), Object.keys(E).sort(), 'competitor set mismatch');
for (const k of Object.keys(E)) close(G[k], E[k], `competitor[${k}]`);

// 2. The dumped projection is conservatively sorted (sanity on the ranking the observer renders).
for (let i = 1; i < leaderboard_projection.length; i++) {
  assert.ok(
    leaderboard_projection[i - 1].conservative >= leaderboard_projection[i].conservative,
    `leaderboard_projection not conservative-sorted at index ${i}`,
  );
}

console.log(`\nPASS  leaderboard parity (JS assembleLeaderboard matches the Python leaderboard() golden; ${got.length} competitors)`);
