// GET /api/creativity - read-only projection of the precomputed creativity snapshot.
//
// Creativity rows are derived from public say-event transcripts by tools/creativity_signal.py.
// The source of truth remains games.transcript_json; this route only serves the latest uploaded
// metric version to the observer.
import { q, send } from './_db.js';
import { roundHalfEven } from './_round.js';

const r3 = (x) => (x == null ? null : roundHalfEven(Number(x), 3));

function boolish(value) {
  return Boolean(Number(value || 0));
}

function intish(value) {
  return Number(value || 0);
}

function publicRole(row) {
  return {
    raw_distance: r3(row.raw_distance),
    role_z: r3(row.role_z),
    valid_utterances: intish(row.valid_utterances),
    valid_pairs: intish(row.valid_pairs),
    sampled_pairs: intish(row.sampled_pairs),
    excluded_pairs: intish(row.excluded_pairs),
    provisional: boolish(row.provisional),
    insufficient: boolish(row.insufficient),
  };
}

export function assembleCreativity(rows, version = null, updatedUtc = null) {
  const byIdentity = {};
  for (const row of rows) {
    const entry = (byIdentity[row.identity_key] ||= {
      version: row.version,
      identity_key: row.identity_key,
      agent_id: row.agent_id,
      display_name: row.display_name,
      declared_model: row.declared_model,
      declared_harness: row.declared_harness,
      by_role: {},
    });

    if (row.role === '*') {
      Object.assign(entry, {
        raw_distance: r3(row.raw_distance),
        overall_z: r3(row.overall_z),
        valid_utterances: intish(row.valid_utterances),
        valid_pairs: intish(row.valid_pairs),
        sampled_pairs: intish(row.sampled_pairs),
        excluded_pairs: intish(row.excluded_pairs),
        eligible_roles: intish(row.eligible_roles),
        provisional: boolish(row.provisional),
        insufficient: boolish(row.insufficient),
        updated_utc: row.updated_utc,
      });
    } else {
      entry.by_role[row.role] = publicRole(row);
    }
  }

  const competitors = Object.values(byIdentity).sort((a, b) => {
    const az = a.overall_z ?? -9999;
    const bz = b.overall_z ?? -9999;
    if (bz !== az) return bz - az;
    const ar = a.raw_distance ?? -9999;
    const br = b.raw_distance ?? -9999;
    if (br !== ar) return br - ar;
    return (b.valid_pairs || 0) - (a.valid_pairs || 0);
  });

  return { version, updated_utc: updatedUtc, competitors };
}

async function creativityTableExists() {
  const rows = await q("SELECT to_regclass('public.creativity_scores') AS table_name");
  return Boolean(rows[0]?.table_name);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  if (req.method === 'GET') {
    try {
      if (!(await creativityTableExists())) {
        return send(res, 200, { version: null, updated_utc: null, competitors: [] });
      }

      const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
      let version = url.searchParams.get('version');
      let updatedUtc = null;
      if (!version) {
        const latest = await q(
          'SELECT version, MAX(updated_utc) AS updated_utc FROM creativity_scores ' +
            'GROUP BY version ORDER BY updated_utc DESC LIMIT 1',
        );
        if (!latest.length) {
          return send(res, 200, { version: null, updated_utc: null, competitors: [] });
        }
        version = latest[0].version;
        updatedUtc = latest[0].updated_utc;
      } else {
        const stamp = await q('SELECT MAX(updated_utc) AS updated_utc FROM creativity_scores WHERE version = $1', [
          version,
        ]);
        updatedUtc = stamp[0]?.updated_utc || null;
      }

      const rows = await q('SELECT * FROM creativity_scores WHERE version = $1', [version]);
      return send(res, 200, assembleCreativity(rows, version, updatedUtc));
    } catch (err) {
      if (err?.code === '42P01') return send(res, 200, { version: null, updated_utc: null, competitors: [] });
      throw err;
    }
  }

  return send(res, 405, { error: 'method not allowed' });
}
