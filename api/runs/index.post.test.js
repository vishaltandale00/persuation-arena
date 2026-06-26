// Codex (final review): POST /api/runs with `connected:true` and `shards>1` must be REJECTED (400),
// NOT silently create a single normal run. Sharded creation isn't wired on the Vercel surface yet
// (parent + K children + K Modal spawns under the diagonal model is host-launcher-only), so a
// `shards` payload here must fail loudly rather than no-op into a normal run.
//
// Drives the REAL handler (mod.default(req,res)) with a mocked _db.js so no DB is opened.
// Run with:  node --experimental-test-module-mocks --test api/runs/index.post.test.js
// (mock.module is experimental.) Flagless => self-skip so a plain `node --test` stays green.
import { test, mock } from 'node:test';
import assert from 'node:assert/strict';

const DB_PATH = new URL('../_db.js', import.meta.url).pathname;
const MODULE_MOCKS = typeof mock.module === 'function';

function fakeReqRes(body) {
  const captured = {};
  const req = { method: 'POST', body };
  const res = {
    statusCode: 200,
    setHeader() {},
    end(b) { captured.body = b ? JSON.parse(b) : {}; captured.status = this.statusCode; },
  };
  return { req, res, captured };
}

async function loadHandlerWithMock(qSink) {
  mock.module(DB_PATH, {
    namedExports: {
      q: async () => { qSink.calls += 1; return []; },
      send: (res, code, body) => { res.statusCode = code; res.end(JSON.stringify(body)); },
      readBody: async (req) => req.body,
      utcnow: () => '2026-01-01T00:00:00Z',
      newId: () => 'abcdef',
      validateUniquePublicNames: () => null,
    },
  });
  return (await import(`./index.js?bust=${qSink.calls}-${Math.random()}`)).default;
}

test('POST /api/runs connected + shards>1 -> 400, no DB write', { skip: MODULE_MOCKS ? false : 'needs --experimental-test-module-mocks' }, async () => {
  const qSink = { calls: 0 };
  const handler = await loadHandlerWithMock(qSink);
  const { req, res, captured } = fakeReqRes({ connected: true, game: 'onuw', players: 5, shards: 2 });
  await handler(req, res);
  assert.equal(captured.status, 400, `expected 400, got ${captured.status} ${JSON.stringify(captured.body)}`);
  assert.match(captured.body.error || '', /shard/i, 'error must explain shards are unsupported here');
  assert.equal(qSink.calls, 0, 'must reject BEFORE any DB write (no silent normal run)');
});
// INV-2 note: the guard is `shards>1`-only, so a shards<=1 / absent payload takes the unchanged
// normal-create path verbatim — covered by the existing handler behavior + the rest of the suite.
