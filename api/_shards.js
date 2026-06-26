// Pure shard-aware read helpers for the Vercel JS port. These mirror the already-implemented Python
// logic so the observer/discovery handlers (api/runs/index.js, api/runs/open.js,
// api/runs/[run_id]/index.js, api/runs/[run_id]/games/[gid].js) present a sharded run the same way
// arena/server.py does over the local FastAPI server.
//
// Reference (read for the contract, do not re-derive):
//   arena/sharded.py        — rollup_parent_status (SPEC §6.4 / REQ-2 / V-2)
//   arena/server.py         — api_runs parent branch, api_run parent branch (_source_run_ids,
//                             _parent_signups, _parent_recent_events), api_game cross-child resolution
//   arena/store.py          — list_open_runs (INV-4 discoverability filter),
//                             create_signup (run_kind/join_token gating, INV-4 / SPEC D7 / INV-2)
//
// Everything here is PURE: it takes plain row objects / arrays (already loaded from Neon by the
// handler) and returns derived values. No DB access, no I/O — so the unit tests feed fake rows.

/** Port of `(run.get("run_kind") or "normal")`: the run's kind, defaulting blank/null to 'normal'. */
export function runKind(run) {
  return (run && run.run_kind) || 'normal';
}

/**
 * Port of arena/server.py _source_run_ids: the run ids whose games/scores/events back a run's
 * observer view (SPEC D7). A normal/child run is backed by itself; a sharded PARENT is backed by its
 * children (pass `childIds` = store.child_run_ids(parent), in shard order). A parent owns no games of
 * its own, so an unstarted parent (no children) yields [] — matching `store.child_run_ids() or []`.
 */
export function sourceRunIds(run, childIds = []) {
  if (runKind(run) === 'parent') return childIds || [];
  return [run.id];
}

/**
 * Port of arena/sharded.py rollup_parent_status: compute a parent's status from its children's
 * statuses (SPEC §6.4 / REQ-2 / V-2). Takes the list of child statuses (a missing/blank status is
 * treated as 'open', matching `(run or {}).get("status") or "open"`).
 *   - no children          -> 'open'
 *   - any child 'partial'  -> 'partial'
 *   - all children 'done'  -> 'done'
 *   - otherwise            -> 'running'
 */
export function rollupParentStatus(childStatuses) {
  if (!childStatuses || childStatuses.length === 0) return 'open';
  const statuses = childStatuses.map((s) => s || 'open');
  if (statuses.some((s) => s === 'partial')) return 'partial';
  if (statuses.every((s) => s === 'done')) return 'done';
  return 'running';
}

/**
 * Port of the arena/server.py api_runs parent branch: the index renders a parent as ONE normal run.
 * Its own row never gets child progress written back, so roll the status up and sum the children's
 * team splits here. Returns { status, teamSplit } for the parent's list entry.
 * `childRuns` are the full child run rows (each with `status` and a `team_split` {good, evil}).
 */
export function aggregateIndexRow(parent, childRuns) {
  const childStatuses = childRuns.map((c) => c.status);
  const teamSplit = { good: 0, evil: 0 };
  for (const c of childRuns) {
    for (const [team, n] of Object.entries(c.team_split || {})) {
      teamSplit[team] = (teamSplit[team] || 0) + (n || 0);
    }
  }
  return { status: rollupParentStatus(childStatuses), teamSplit };
}

/**
 * Port of the arena/server.py api_run parent branch aggregate fields: source EVERY aggregate from the
 * children (the parent row owns no games/wins/team-split of its own). Children carry disjoint global
 * gids (D8), so the games union is just concat-then-sort-by-gid with no de-dup, and per-name wins /
 * team-split sum cleanly. Returns { games, wins, team_split, status }.
 * `childRuns` are the full child run rows (each with `games` [{gid,...}], `wins` {name:n},
 * `team_split` {good,evil}, `status`).
 */
export function aggregateParentDetail(parent, childRuns) {
  const games = parentGameDisplayRows(childRuns);
  const wins = {};
  const teamSplit = { good: 0, evil: 0 };
  for (const cr of childRuns) {
    for (const [name, w] of Object.entries(cr.wins || {})) {
      wins[name] = (wins[name] || 0) + (w || 0);
    }
    for (const [team, n] of Object.entries(cr.team_split || {})) {
      teamSplit[team] = (teamSplit[team] || 0) + (n || 0);
    }
  }
  return {
    games,
    wins,
    team_split: teamSplit,
    status: rollupParentStatus(childRuns.map((c) => c.status)),
  };
}

/**
 * Parent game rows for the observer. Normal sharded children have disjoint global gids, so preserve
 * them. Manual rollups of older independent runs can reuse gids; in that case assign display gids in
 * child order while retaining the source run/gid for transcript lookup.
 */
export function parentGameDisplayRows(childRuns) {
  const seen = new Set();
  let hasDuplicate = false;
  const byChild = childRuns.map((cr) => ({
    id: cr.id,
    games: [...(cr.games || [])].sort((a, b) => a.gid - b.gid),
  }));
  for (const cr of byChild) {
    for (const g of cr.games) {
      const key = String(g.gid);
      if (seen.has(key)) hasDuplicate = true;
      seen.add(key);
    }
  }

  if (!hasDuplicate) {
    const rows = [];
    for (const cr of byChild) {
      for (const g of cr.games) rows.push({ ...g, source_run_id: cr.id, source_gid: g.gid });
    }
    rows.sort((a, b) => a.gid - b.gid);
    return rows;
  }

  const rows = [];
  let displayGid = 1;
  for (const cr of byChild) {
    for (const g of cr.games) {
      rows.push({ ...g, source_run_id: cr.id, source_gid: g.gid, gid: displayGid });
      displayGid += 1;
    }
  }
  return rows;
}

export function parentGameSource(childRuns, displayGid) {
  const target = Number(displayGid);
  const row = parentGameDisplayRows(childRuns).find((g) => Number(g.gid) === target);
  if (!row) return null;
  return { runId: row.source_run_id, gid: row.source_gid };
}

// Status rank for collapsing a sharded parent's per-shard signups to one logical signup per agent
// (arena/server.py _parent_signups). expired/cancelled rank below waiting so a live shard always
// outranks a dead one.
const SIGNUP_STATUS_RANK = {
  waiting: 0, ready_required: 1, ready: 2, active: 3, completed: 4, expired: -1, cancelled: -1,
};

/**
 * Port of arena/server.py _parent_signups: the children share identities (REQ-5), so the same agent
 * is signed up on every shard. Present ONE logical signup per agent_id, picking the most-advanced
 * status across shards so the roster row reflects whether that competitor has reached
 * ready/active/completed anywhere. `signupsByChild` is an array (one entry per child) of that child's
 * signup rows.
 */
export function parentSignups(signupsByChild) {
  const best = new Map();
  for (const childSignups of signupsByChild) {
    for (const s of childSignups) {
      const cur = best.get(s.agent_id);
      const rank = (st) => (SIGNUP_STATUS_RANK[st] ?? 0);
      if (!cur || rank(s.status) > rank(cur.status)) best.set(s.agent_id, s);
    }
  }
  return Array.from(best.values());
}

/**
 * Port of arena/server.py _parent_recent_events: union of the children's recent events, globally
 * ordered (created_utc, then per-shard seq) and capped like a single run's stream (last 120). The
 * parent owns no events of its own. `eventsByChild` is an array (one entry per child) of that child's
 * already-shaped event rows (each carrying created_utc and seq).
 */
export function parentRecentEvents(eventsByChild) {
  const events = [];
  for (const childEvents of eventsByChild) for (const e of childEvents) events.push(e);
  events.sort((a, b) => {
    const ca = a.created_utc || '';
    const cb = b.created_utc || '';
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    return (a.seq || 0) - (b.seq || 0);
  });
  return events.slice(-120);
}

/**
 * Port of the arena/store.py list_open_runs INV-4 filter: only run_kind=='normal' runs are publicly
 * discoverable/joinable; shard parents and children are excluded from the open-runs list.
 */
export function isDiscoverableOpenRun(run) {
  return runKind(run) === 'normal';
}

/**
 * Port of the arena/store.py create_signup run_kind/join_token gate (INV-4 / SPEC D7 / INV-2).
 * Returns an error reason string ('run_not_joinable') or null when joining is permitted:
 *   - parent: NEVER joinable (presentational umbrella) -> 'run_not_joinable'
 *   - child:  joinable ONLY when the run's join_token is set AND matches the supplied token
 *   - normal: ignores the token entirely -> null (byte-identical to pre-shard behavior)
 */
export function signupGateError(run, joinToken) {
  const kind = runKind(run);
  if (kind === 'parent') return 'run_not_joinable';
  if (kind === 'child') {
    const expected = run.join_token || null;
    if (!expected || joinToken !== expected) return 'run_not_joinable';
  }
  return null;
}
