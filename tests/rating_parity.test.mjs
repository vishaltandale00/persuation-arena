// Parity test: api/_rating.js computeRatings() vs the Python golden snapshot.
// Run:  node tests/rating_parity.test.mjs
//
// Loads tests/fixtures/rating_golden.json, runs the pure JS computeRatings() on the fixture INPUTS,
// and asserts the three outputs match the fixture OUTPUTS (the Python arena.rating.recompute()
// snapshot): elo/skill/rd/floats within 1e-6, games/wins/forfeit counts exact, identity_keys and
// the full ordering exact.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// _rating.js transitively imports _db.js which calls neon(process.env.DATABASE_URL) at module load.
// computeRatings() itself never touches the DB, but the import needs a non-empty URL.
process.env.DATABASE_URL ||= 'postgres://dummy:dummy@localhost:5432/dummy';

const { computeRatings } = await import('../api/_rating.js');

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'rating_golden.json'), 'utf8'));

const inputs = {
  gamePlayers: fixture.inputs.all_game_player_rows,
  runMeta: fixture.inputs.run_meta_map,
  agents: fixture.inputs.list_agents,
};

const expected = fixture.outputs;
const got = computeRatings(inputs);

const EPS = 1e-6;
const FLOAT_FIELDS = new Set([
  'skill', 'rd', 'elo', 'forfeit_rate', // ratings
  'pre_skill', 'post_skill', 'delta', 'expected', 'd_r', 'resistance', 'k', 'damped', // events
  'base_rate', // role_difficulty
]);

function approx(label, a, b) {
  assert.ok(Math.abs(a - b) <= EPS, `${label}: ${a} != ${b} (|Δ|=${Math.abs(a - b)} > ${EPS})`);
}

function compareRow(label, gotRow, expRow) {
  const keys = Object.keys(expRow);
  for (const key of keys) {
    const e = expRow[key];
    const g = gotRow[key];
    if (FLOAT_FIELDS.has(key) && typeof e === 'number' && typeof g === 'number') {
      approx(`${label}.${key}`, g, e);
    } else {
      assert.deepEqual(g, e, `${label}.${key}: ${JSON.stringify(g)} != ${JSON.stringify(e)}`);
    }
  }
}

// --- 1. role_difficulty (sorted list of cells) ---------------------------------------------------
{
  const exp = expected.role_difficulty_map;
  const g = got.roleDifficulty;
  assert.equal(g.length, exp.length, `role_difficulty count: ${g.length} != ${exp.length}`);
  for (let i = 0; i < exp.length; i++) {
    // ordering must match exactly (bucket,role)
    assert.equal(g[i].bucket, exp[i].bucket, `role_difficulty[${i}].bucket order`);
    assert.equal(g[i].role, exp[i].role, `role_difficulty[${i}].role order`);
    compareRow(`role_difficulty[${i}]`, g[i], exp[i]);
  }
  console.log(`ok  role_difficulty: ${g.length} cells`);
}

// --- 2. rating_events (sorted by created_utc,run_id,gid,seat) -------------------------------------
{
  const exp = expected.all_rating_events;
  const g = got.ratingEvents;
  assert.equal(g.length, exp.length, `rating_events count: ${g.length} != ${exp.length}`);
  for (let i = 0; i < exp.length; i++) {
    assert.equal(g[i].id, exp[i].id, `rating_events[${i}].id order: ${g[i].id} != ${exp[i].id}`);
    compareRow(`rating_events[${i}]`, g[i], exp[i]);
  }
  console.log(`ok  rating_events: ${g.length} events`);
}

// --- 3. ratings vs leaderboard_rows --------------------------------------------------------------
// The fixture's leaderboard_rows is the recompute()-built ratings list, dumped sorted by
// identity_key (the generator's stable-diff sort) — which is exactly the order computeRatings emits
// (sorted(skills.keys())). So the order must match positionally, AND every per-row field.
{
  const exp = expected.leaderboard_rows;
  const g = got.ratings;
  assert.equal(g.length, exp.length, `ratings count: ${g.length} != ${exp.length}`);

  assert.deepEqual(
    g.map((r) => r.identity_key),
    exp.map((r) => r.identity_key),
    'ratings identity_key ordering mismatch (expected sorted(identity_key))',
  );

  for (let i = 0; i < exp.length; i++) {
    compareRow(`ratings[${exp[i].identity_key}]`, g[i], exp[i]);
  }

  // sanity: the prod read path sorts by the conservative lower bound; verify it's well-defined and
  // reproducible from our stored rows (no field is missing for the SQL ORDER BY).
  const ELO_SCALE = 173.0;
  const conservative = (r) => r.elo - 2 * ELO_SCALE * r.rd;
  for (const r of g) assert.ok(Number.isFinite(conservative(r)), `non-finite conservative for ${r.identity_key}`);

  console.log(`ok  ratings: ${g.length} competitors`);
}

// --- 4. summary matches the recompute() summary --------------------------------------------------
{
  const exp = fixture._meta.recompute_summary;
  assert.deepEqual(got.summary, exp, `summary mismatch: ${JSON.stringify(got.summary)} != ${JSON.stringify(exp)}`);
  console.log(`ok  summary: ${JSON.stringify(got.summary)}`);
}

console.log('\nPASS  rating parity (JS computeRatings matches the Python golden snapshot)');
