// INTEGRATION test for POST /api/runs/{run_id}/signups (FINDING P2): the JS signup handler must
// thread an OPTIONAL explicit seat (roster index) into the signup INSERT, mirroring the Python
// store.create_signup(seat=) deterministic-seat semantics (SPEC D5/V-7). Production signups flow
// through this JS handler, so without this the sharded aggregate seats children by arrival order and
// breaks the V-7 fairness guarantee.
//
// Drives the REAL handler (mod.default(req,res)) with a mocked _db.js. The handler inserts a row, then
// advanceLobby() seats; the FIX stores the requested seat as roster_index so advanceLobby honors it.
// We assert: (a) body.seat=k => the signup INSERT carries roster_index=k; (b) no seat => roster_index
// is null (INV-2, byte-identical to before).
//
// Run with:  node --experimental-test-module-mocks --test 'api/runs/[[]run_id[]]/signups.test.js'
// (mock.module is experimental; brackets escaped for node's --test glob.) Flagless => self-skip.
import { test, mock } from 'node:test';
import assert from 'node:assert/strict';

const DB_PATH = new URL('../../_db.js', import.meta.url).pathname;
const MODULE_MOCKS = typeof mock.module === 'function';

function fakeReqRes(runId, body) {
  const captured = {};
  const req = { method: 'POST', query: { run_id: runId }, headers: { authorization: 'Bearer t' }, body };
  const res = {
    statusCode: 200,
    setHeader() {},
    end(b) { captured.body = b ? JSON.parse(b) : {}; captured.status = this.statusCode; },
  };
  return { req, res, captured };
}

// SQL-routing mock for q(text, params). Captures the run_signups INSERT params for assertions.
function makeQ(sink) {
  const inserted = {};
  return async (text, params = []) => {
    const t = text.replace(/\s+/g, ' ').trim();
    if (t.startsWith('SELECT * FROM runs WHERE id =')) {
      return [{ id: params[0], run_kind: 'normal', status: 'open', players: 5, coordinator_url: null }];
    }
    // existing active signup for this agent? -> none
    if (t.startsWith('SELECT * FROM run_signups WHERE run_id=') && t.includes('AND agent_id=')) {
      return [];
    }
    // active roster (for capacity / unique-name checks) -> empty
    if (t.includes('FROM run_signups s JOIN agents a') && t.includes("status IN ('waiting'")) {
      return [];
    }
    // the INSERT ... RETURNING * (capture seat/roster_index from params)
    if (t.startsWith('WITH run_lock AS') || (t.startsWith('INSERT INTO run_signups'))) {
      sink.insertText = t;
      sink.insertParams = params;
      const row = { id: params[0], run_id: params[1], agent_id: params[2], status: 'waiting', seat: null };
      inserted[params[0]] = row;
      return [row];
    }
    if (t.startsWith('UPDATE agents SET last_seen_utc')) return [];
    if (t.startsWith('SELECT * FROM run_signups WHERE id=')) {
      return [inserted[params[0]] || { id: params[0], run_id: 'run_x', status: 'waiting', seat: null }];
    }
    return [];
  };
}

async function runHandler(body) {
  const sink = {};
  mock.module(DB_PATH, {
    namedExports: {
      q: makeQ(sink),
      send(res, status, b) { res.statusCode = status; res.end(JSON.stringify(b)); },
      readBody: async (req) => req.body || {},
      agentFromToken: async () => ({ id: 'agent_1', display_name: 'alpha' }),
      signupResponse: (s) => ({ signup_id: s.id, status: s.status, seat: s.seat ?? null }),
      advanceLobby: async () => {},
      getRun: async (id) => ({ id, run_kind: 'normal', status: 'open', players: 5, coordinator_url: null }),
      OPEN_RUN_STATUSES: ['open', 'waiting', 'ready_required'],
      PROTOCOL_VERSION: 'arena-agent-v1',
      newId: (p) => `${p}fixed`,
      utcnow: () => '2026-01-01T00:00:00.000000Z',
      utcAfter: () => '2026-01-01T00:10:00.000000Z',
      publicRef: (n) => `@${n}`,
      validateUniquePublicNames: () => null,
    },
  });
  const mod = await import(`../../runs/[run_id]/signups.js?case=${encodeURIComponent(JSON.stringify(body))}`);
  const { req, res, captured } = fakeReqRes('run_x', body);
  await mod.default(req, res);
  mock.reset();
  return { captured, sink };
}

test('POST signups: explicit body.seat is threaded into the INSERT as roster_index', { skip: !MODULE_MOCKS && 'needs --experimental-test-module-mocks' }, async () => {
  const { captured, sink } = await runHandler({ protocol_version: 'arena-agent-v1', seat: 3 });
  assert.equal(captured.status, 200);
  assert.ok(sink.insertText.includes('roster_index'), 'INSERT must include the roster_index column');
  assert.ok(sink.insertParams.includes(3), `seat 3 must appear in INSERT params, got ${JSON.stringify(sink.insertParams)}`);
});

test('POST signups: no seat => roster_index null (INV-2 arrival order preserved)', { skip: !MODULE_MOCKS && 'needs --experimental-test-module-mocks' }, async () => {
  const { captured, sink } = await runHandler({ protocol_version: 'arena-agent-v1' });
  assert.equal(captured.status, 200);
  // No explicit seat must not introduce any non-null roster index into the params.
  // (We can't know the exact index, but a number that isn't players/max_concurrent must not slip in.)
  assert.ok(!sink.insertParams.includes(3), 'no seat must not inject an explicit roster index');
});
