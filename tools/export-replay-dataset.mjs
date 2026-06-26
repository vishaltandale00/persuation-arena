#!/usr/bin/env node
import crypto from 'node:crypto';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { pathToFileURL } from 'node:url';

import { neon } from '@neondatabase/serverless';

import {
  REPLAY_SCHEMA_VERSION,
  buildReplayDataset,
  zipReplayDataset,
} from './replay_dataset_export.mjs';

function usage() {
  return `Usage:
  node tools/export-replay-dataset.mjs --run glm_demo_2 --out /tmp/pa-replay.zip
  node tools/export-replay-dataset.mjs --run glm_demo_2 --upload

Options:
  --run <run_id>       Run id to export. May be repeated or comma-separated.
  --out <path>         Local zip path. Defaults to /tmp/pa-replay-<timestamp>.zip.
  --upload             Upload private zip and manifest to Vercel Blob.
  --blob-path <path>   Blob zip pathname. Defaults to datasets/pa-replay-v0.1/<timestamp>/replay.zip.
  --help               Show this help.
`;
}

function timestampSlug(createdUtc) {
  return createdUtc.replace(/\.\d{3}Z$/, 'Z').replace(/[:.]/g, '-');
}

export function parseArgs(argv) {
  const opts = { runIds: [], upload: false, out: null, blobPath: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
    } else if (arg === '--upload') {
      opts.upload = true;
    } else if (arg === '--run') {
      const value = argv[++i];
      if (!value) throw new Error('--run requires a value');
      opts.runIds.push(...value.split(',').map((v) => v.trim()).filter(Boolean));
    } else if (arg === '--out') {
      opts.out = argv[++i];
      if (!opts.out) throw new Error('--out requires a value');
    } else if (arg === '--blob-path') {
      opts.blobPath = argv[++i];
      if (!opts.blobPath) throw new Error('--blob-path requires a value');
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  opts.runIds = [...new Set(opts.runIds)];
  return opts;
}

async function query(sql, text, params = []) {
  return await sql.query(text, params);
}

export async function fetchRowsForRuns(sql, runIds) {
  if (!runIds.length) throw new Error('at least one --run is required');

  const runs = await query(
    sql,
    `SELECT * FROM runs
      WHERE id = ANY($1::text[]) AND status = 'done'
      ORDER BY created_utc, id`,
    [runIds],
  );
  const foundRunIds = runs.map((r) => r.id);
  const missing = runIds.filter((id) => !foundRunIds.includes(id));
  if (missing.length) {
    throw new Error(`no completed run found for: ${missing.join(', ')}`);
  }

  const games = await query(
    sql,
    `SELECT * FROM games
      WHERE run_id = ANY($1::text[])
      ORDER BY run_id, gid`,
    [foundRunIds],
  );
  const gamePlayers = await query(
    sql,
    `SELECT * FROM game_players
      WHERE run_id = ANY($1::text[])
      ORDER BY run_id, gid, seat`,
    [foundRunIds],
  );
  const runEvents = await query(
    sql,
    `SELECT * FROM run_events
      WHERE run_id = ANY($1::text[])
      ORDER BY run_id, seq`,
    [foundRunIds],
  );
  const turns = await query(
    sql,
    `SELECT * FROM turns
      WHERE run_id = ANY($1::text[])
      ORDER BY run_id, game_instance_id, created_utc, id`,
    [foundRunIds],
  );
  const turnIds = turns.map((t) => t.id);
  const turnReplies = turnIds.length
    ? await query(
      sql,
      `SELECT * FROM turn_replies
        WHERE turn_id = ANY($1::text[])
        ORDER BY created_utc, turn_id`,
      [turnIds],
    )
    : [];
  const runSignups = await query(
    sql,
    `SELECT s.*, a.display_name, a.declared_model, a.declared_harness
      FROM run_signups s
      LEFT JOIN agents a ON a.id = s.agent_id
      WHERE s.run_id = ANY($1::text[])
      ORDER BY s.run_id, s.seat IS NULL, s.seat, s.created_utc`,
    [foundRunIds],
  );

  const agentIds = [...new Set([
    ...runSignups.map((s) => s.agent_id),
    ...gamePlayers.map((p) => p.agent_id),
  ].filter(Boolean))];
  const agents = agentIds.length
    ? await query(
      sql,
      `SELECT id, display_name, protocol_version, sdk_version, created_utc,
              last_seen_utc, status, declared_model, declared_harness
        FROM agents
        WHERE id = ANY($1::text[])
        ORDER BY id`,
      [agentIds],
    )
    : [];

  return {
    runs,
    games,
    gamePlayers,
    runEvents,
    turns,
    turnReplies,
    agents,
    runSignups,
  };
}

async function uploadToBlob({ zipBuffer, manifest, blobPath, token }) {
  if (!token) throw new Error('BLOB_READ_WRITE_TOKEN is required for --upload');
  const { put } = await import('@vercel/blob');
  const manifestPath = blobPath.replace(/\/[^/]*$/, '/manifest.json');
  const common = {
    access: 'private',
    addRandomSuffix: false,
    token,
  };
  const zip = await put(blobPath, zipBuffer, {
    ...common,
    contentType: 'application/zip',
  });
  const manifestUpload = await put(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, {
    ...common,
    contentType: 'application/json',
  });
  return { zip, manifest: manifestUpload };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    process.stdout.write(usage());
    return;
  }
  if (!opts.runIds.length) throw new Error('at least one --run is required');
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is required to query Neon');
  }

  const createdUtc = new Date().toISOString();
  const slug = timestampSlug(createdUtc);
  const outPath = opts.out || `/tmp/pa-replay-${slug}.zip`;
  const blobPath = opts.blobPath || `datasets/${REPLAY_SCHEMA_VERSION}/${slug}/replay.zip`;

  const sql = neon(process.env.DATABASE_URL);
  const rows = await fetchRowsForRuns(sql, opts.runIds);
  const dataset = buildReplayDataset(rows, {
    createdUtc,
    runIds: opts.runIds,
    sourceKind: 'neon',
    sourceDatabase: 'DATABASE_URL environment',
    statuses: ['done'],
  });
  const zipBuffer = zipReplayDataset(dataset.files);
  const sha256 = crypto.createHash('sha256').update(zipBuffer).digest('hex');

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, zipBuffer);

  let blob = null;
  if (opts.upload) {
    blob = await uploadToBlob({
      zipBuffer,
      manifest: dataset.manifest,
      blobPath,
      token: process.env.BLOB_READ_WRITE_TOKEN,
    });
  }

  process.stdout.write(`${JSON.stringify({
    ok: true,
    out: outPath,
    bytes: zipBuffer.length,
    sha256,
    counts: dataset.manifest.counts,
    files: {
      games: dataset.manifest.files.games.length,
      perspectives: dataset.manifest.files.perspectives.length,
    },
    blob,
  }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    process.stderr.write(`${err?.stack || err?.message || String(err)}\n`);
    process.exitCode = 1;
  });
}
