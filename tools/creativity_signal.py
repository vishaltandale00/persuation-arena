"""Compute and upload the role-conditioned creativity signal.

The metric is derived from public transcript `say` events:
  same identity + same dealt role -> pair public utterances -> judge similarity -> distance.

This script is intentionally separate from the game runner. It never changes games, players, ratings,
or rating events. Uploads are additive/upsert-only derived snapshots.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


TOKEN_RE = re.compile(r"[A-Za-z0-9_@']+")
VALID_COHERENCE = {"valid", "underinformative", "off_task"}
PROMPT_VERSION = "creativity_judge_v1"
EMBEDDING_MODEL = "local_tfidf_v0"

SCORES_DDL = """
CREATE TABLE IF NOT EXISTS creativity_scores (
  version TEXT, identity_key TEXT, agent_id TEXT, display_name TEXT,
  declared_model TEXT, declared_harness TEXT, role TEXT,
  raw_distance DOUBLE PRECISION, role_z DOUBLE PRECISION, overall_z DOUBLE PRECISION,
  valid_utterances INTEGER, valid_pairs INTEGER, sampled_pairs INTEGER,
  excluded_pairs INTEGER, eligible_roles INTEGER,
  provisional INTEGER, insufficient INTEGER, updated_utc TEXT,
  PRIMARY KEY (version, identity_key, role)
)
"""

JUDGMENTS_DDL = """
CREATE TABLE IF NOT EXISTS creativity_judgments (
  version TEXT, judgment_key TEXT PRIMARY KEY, identity_key TEXT, role TEXT,
  utterance_a TEXT, utterance_b TEXT,
  embedding_model TEXT, embedding_similarity DOUBLE PRECISION,
  judge_model TEXT, judge_temperature DOUBLE PRECISION, judge_prompt_version TEXT,
  judge_similarity DOUBLE PRECISION, distance DOUBLE PRECISION,
  coherence_a TEXT, coherence_b TEXT,
  reason TEXT, divergence_phrases_json TEXT,
  tokens_a INTEGER, tokens_b INTEGER, length_ratio DOUBLE PRECISION, created_utc TEXT
)
"""

SCORE_COLS = [
    "version", "identity_key", "agent_id", "display_name", "declared_model", "declared_harness",
    "role", "raw_distance", "role_z", "overall_z", "valid_utterances", "valid_pairs",
    "sampled_pairs", "excluded_pairs", "eligible_roles", "provisional", "insufficient", "updated_utc",
]

JUDGMENT_COLS = [
    "version", "judgment_key", "identity_key", "role", "utterance_a", "utterance_b",
    "embedding_model", "embedding_similarity", "judge_model", "judge_temperature",
    "judge_prompt_version", "judge_similarity", "distance", "coherence_a", "coherence_b", "reason",
    "divergence_phrases_json", "tokens_a", "tokens_b", "length_ratio", "created_utc",
]


@dataclass(frozen=True)
class Identity:
    identity_key: str
    agent_id: str | None
    display_name: str
    declared_model: str | None
    declared_harness: str | None


@dataclass
class Utterance:
    id: str
    identity: Identity
    role: str
    text: str
    masked_text: str
    vector: dict[str, float] | None = None


@dataclass
class Pair:
    id: str
    identity: Identity
    role: str
    a: Utterance
    b: Utterance
    embedding_similarity: float


def utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def load_env(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def public_ref(name: str) -> str:
    handle = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return f"@{handle}" if handle else ""


def mask_names(text: str, names: list[str]) -> str:
    out = text or ""
    refs = {public_ref(name) for name in names if public_ref(name)}
    for ref in sorted(refs, key=len, reverse=True):
        out = re.sub(re.escape(ref) + r"\b", "@player", out, flags=re.IGNORECASE)
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        out = re.sub(rf"(?<![\w@]){re.escape(name)}(?![\w-])", "Player", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return max(0.0, min(1.0, sum(v * b.get(k, 0.0) for k, v in a.items())))


def attach_tfidf(utterances: list[Utterance]) -> None:
    df: Counter[str] = Counter()
    for u in utterances:
        df.update(set(tokens(u.masked_text)))
    n = max(1, len(utterances))
    for u in utterances:
        tf = Counter(tokens(u.masked_text))
        vec: dict[str, float] = {}
        for term, count in tf.items():
            vec[term] = (1 + math.log(count)) * math.log((n + 1) / (df[term] + 1)) + 1
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        u.vector = {k: v / norm for k, v in vec.items()}


def identity_for(row: sqlite3.Row, roster_harness: dict[tuple[str, str], str],
                 agents_by_id: dict[str, dict[str, Any]]) -> Identity:
    agent_id = row["agent_id"]
    if agent_id:
        agent = agents_by_id.get(agent_id, {})
        return Identity(
            identity_key=agent_id,
            agent_id=agent_id,
            display_name=agent.get("display_name") or row["agent"] or agent_id,
            declared_model=agent.get("declared_model"),
            declared_harness=agent.get("declared_harness"),
        )
    model = row["model"] or "?"
    if model == "connected-agent":
        name = row["agent"] or f"seat-{row['seat']}"
        return Identity(
            identity_key=f"legacy-connected:{row['run_id']}:{name}",
            agent_id=None,
            display_name=name,
            declared_model=model,
            declared_harness="connected-legacy",
        )
    harness = roster_harness.get((row["run_id"], row["agent"]), "base")
    return Identity(
        identity_key=f"static:{model}:{harness}",
        agent_id=None,
        display_name=model,
        declared_model=model,
        declared_harness=harness,
    )


def load_utterances(sqlite_path: Path) -> list[Utterance]:
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    try:
        run_rows = con.execute("SELECT id, agents_json FROM runs").fetchall()
        roster_harness: dict[tuple[str, str], str] = {}
        for run in run_rows:
            try:
                agents = json.loads(run["agents_json"] or "[]")
            except (TypeError, ValueError):
                agents = []
            for agent in agents:
                if agent.get("name"):
                    roster_harness[(run["id"], agent["name"])] = agent.get("harness") or "base"

        agents_by_id = {
            r["id"]: dict(r) for r in con.execute(
                "SELECT id, display_name, declared_model, declared_harness FROM agents"
            ).fetchall()
        }
        gp = {
            (r["run_id"], int(r["gid"]), int(r["seat"])): r for r in con.execute(
                "SELECT run_id,gid,seat,agent,agent_id,model,dealt_role FROM game_players"
            ).fetchall()
        }
        out: list[Utterance] = []
        for game in con.execute("SELECT run_id,gid,transcript_json FROM games ORDER BY run_id,gid").fetchall():
            transcript = json.loads(game["transcript_json"])
            names = [p.get("name") for p in transcript.get("players", []) if p.get("name")]
            say_idx = 0
            for phase in transcript.get("phases", []):
                for event in phase.get("events", []):
                    if event.get("t") != "say":
                        continue
                    pid = int(event["pid"])
                    row = gp.get((game["run_id"], int(game["gid"]), pid))
                    if not row:
                        continue
                    text = " ".join((event.get("text") or "").split())
                    if not text:
                        continue
                    ident = identity_for(row, roster_harness, agents_by_id)
                    out.append(Utterance(
                        id=f"{game['run_id']}:{game['gid']}:{say_idx}",
                        identity=ident,
                        role=row["dealt_role"] or "?",
                        text=text,
                        masked_text=mask_names(text, names),
                    ))
                    say_idx += 1
        return out
    finally:
        con.close()


def select_pairs(utterances: list[Utterance], max_pairs_per_bucket: int,
                 limit_buckets: int | None = None) -> tuple[list[Pair], dict[tuple[str, str], list[Utterance]]]:
    attach_tfidf(utterances)
    buckets: dict[tuple[str, str], list[Utterance]] = defaultdict(list)
    for u in utterances:
        buckets[(u.identity.identity_key, u.role)].append(u)

    ordered_buckets = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    if limit_buckets:
        ordered_buckets = ordered_buckets[:limit_buckets]

    pairs: list[Pair] = []
    kept_buckets: dict[tuple[str, str], list[Utterance]] = {}
    for key, items in ordered_buckets:
        if len(items) < 2:
            continue
        scored: list[tuple[float, str, Utterance, Utterance]] = []
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                sim = cosine(a.vector or {}, b.vector or {})
                scored.append((1.0 - sim, f"{a.id}|{b.id}", a, b))
        scored.sort(key=lambda x: (x[0], x[1]))
        if not scored:
            continue
        if len(scored) <= max_pairs_per_bucket:
            selected = scored
        elif max_pairs_per_bucket <= 1:
            selected = [scored[len(scored) // 2]]
        else:
            idxs = sorted({round(i * (len(scored) - 1) / (max_pairs_per_bucket - 1))
                           for i in range(max_pairs_per_bucket)})
            selected = [scored[i] for i in idxs]
        kept_buckets[key] = items
        for _, _, a, b in selected:
            pairs.append(Pair(
                id=f"p{len(pairs):04d}",
                identity=a.identity,
                role=a.role,
                a=a,
                b=b,
                embedding_similarity=cosine(a.vector or {}, b.vector or {}),
            ))
    return pairs, kept_buckets


def pair_hash(version: str, pair: Pair, judge_model: str, judge_temperature: float | None) -> str:
    payload = {
        "version": version,
        "identity_key": pair.identity.identity_key,
        "role": pair.role,
        "utterance_a": pair.a.masked_text,
        "utterance_b": pair.b.masked_text,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_similarity": round(pair.embedding_similarity, 6),
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "judge_prompt_version": PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def clip01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def norm_coherence(value: Any) -> str:
    v = str(value or "valid").strip().lower()
    return v if v in VALID_COHERENCE else "valid"


def judge_pairs(pairs: list[Pair], *, env: dict[str, str], judge_model: str, judge_temperature: float | None,
                batch_size: int, version: str) -> list[dict[str, Any]]:
    key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the judge pass")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    system = (
        "Judge same-agent, same-role public speech pairs for linguistic self-variety. "
        "Use only the two utterances, role, and embedding_similarity_prior. The prior is a weak hint, "
        "not ground truth. similarity is continuous from 0.0 to 1.0: 1.0 means same wording or same "
        "rhetorical/semantic move; 0.0 means a very different coherent persuasive approach. "
        "Do not reward length alone. Mark each utterance coherence as valid, underinformative, or off_task. "
        "None, pass-only, one-token, vote-only, empty, or generic filler should be underinformative unless "
        "there is enough persuasive content. Return strict JSON only: "
        "{\"pairs\":[{\"id\":\"p0000\",\"similarity\":0.0,\"coherence_a\":\"valid\","
        "\"coherence_b\":\"valid\",\"reason\":\"short\",\"divergence_phrases\":[\"short\",\"short\"]}]}."
    )

    out: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        pending = {p.id: p for p in batch}
        for attempt in range(1, 4):
            payload = {"pairs": [{
                "id": p.id,
                "role": p.role,
                "embedding_similarity_prior": round(p.embedding_similarity, 3),
                "utterance_a": p.a.masked_text,
                "utterance_b": p.b.masked_text,
            } for p in pending.values()]}
            request = {
                "model": judge_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "response_format": {"type": "json_object"},
                "extra_headers": {
                    "HTTP-Referer": "https://persuation-arena.vercel.app",
                    "X-Title": "Persuasion Arena creativity signal",
                },
            }
            if judge_temperature is not None:
                request["temperature"] = judge_temperature
            resp = client.chat.completions.create(**request)
            data = json.loads(resp.choices[0].message.content or "{}")
            for item in data.get("pairs", []):
                p = pending.pop(item.get("id"), None)
                if not p:
                    continue
                sim = clip01(item.get("similarity"))
                ta, tb = len(tokens(p.a.masked_text)), len(tokens(p.b.masked_text))
                ratio = (max(ta, tb) / max(1, min(ta, tb))) if max(ta, tb) else None
                phrases = item.get("divergence_phrases") or []
                out.append({
                    "version": version,
                    "judgment_key": pair_hash(version, p, judge_model, judge_temperature),
                    "identity_key": p.identity.identity_key,
                    "role": p.role,
                    "utterance_a": p.a.masked_text,
                    "utterance_b": p.b.masked_text,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_similarity": p.embedding_similarity,
                    "judge_model": judge_model,
                    "judge_temperature": judge_temperature,
                    "judge_prompt_version": PROMPT_VERSION,
                    "judge_similarity": sim,
                    "distance": 1.0 - sim,
                    "coherence_a": norm_coherence(item.get("coherence_a")),
                    "coherence_b": norm_coherence(item.get("coherence_b")),
                    "reason": str(item.get("reason") or "")[:400],
                    "divergence_phrases_json": json.dumps(phrases[:4], ensure_ascii=False),
                    "tokens_a": ta,
                    "tokens_b": tb,
                    "length_ratio": ratio,
                    "created_utc": utcnow(),
                    "_pair_id": p.id,
                    "_utterance_a_id": p.a.id,
                    "_utterance_b_id": p.b.id,
                    "_identity": p.identity,
                })
            if not pending:
                break
            print(
                f"retrying {len(pending)} missing pair(s) from batch {start // batch_size + 1} "
                f"(attempt {attempt})",
                flush=True,
            )
        if pending:
            missing = ", ".join(sorted(pending))
            raise RuntimeError(f"judge omitted pair ids after retries: {missing}")
        print(f"judged {min(start + batch_size, len(pairs))}/{len(pairs)} pairs", flush=True)
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def compute_scores(judgments: list[dict[str, Any]], kept_buckets: dict[tuple[str, str], list[Utterance]], *,
                   version: str, min_valid_utterances: int, min_valid_pairs: int,
                   min_field_agents: int, min_eligible_roles: int, role_weight_cap: int) -> list[dict[str, Any]]:
    stamp = utcnow()
    stats: dict[tuple[str, str], dict[str, Any]] = {
        key: {
            "identity": items[0].identity,
            "role": key[1],
            "valid_utterances": set(),
            "distances": [],
            "sampled_pairs": 0,
            "excluded_pairs": 0,
        }
        for key, items in kept_buckets.items()
    }

    for j in judgments:
        key = (j["identity_key"], j["role"])
        s = stats.get(key)
        if not s:
            continue
        s["sampled_pairs"] += 1
        if j["coherence_a"] == "valid":
            s["valid_utterances"].add(j["_utterance_a_id"])
        if j["coherence_b"] == "valid":
            s["valid_utterances"].add(j["_utterance_b_id"])
        if j["coherence_a"] == "valid" and j["coherence_b"] == "valid":
            s["distances"].append(float(j["distance"]))
        else:
            s["excluded_pairs"] += 1

    role_rows: list[dict[str, Any]] = []
    by_role_valid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (identity_key, role), s in stats.items():
        ident: Identity = s["identity"]
        valid_utterances = len(s["valid_utterances"])
        valid_pairs = len(s["distances"])
        raw = mean(s["distances"])
        provisional = valid_utterances < min_valid_utterances or valid_pairs < min_valid_pairs
        row = {
            "version": version,
            "identity_key": identity_key,
            "agent_id": ident.agent_id,
            "display_name": ident.display_name,
            "declared_model": ident.declared_model,
            "declared_harness": ident.declared_harness,
            "role": role,
            "raw_distance": raw,
            "role_z": None,
            "overall_z": None,
            "valid_utterances": valid_utterances,
            "valid_pairs": valid_pairs,
            "sampled_pairs": s["sampled_pairs"],
            "excluded_pairs": s["excluded_pairs"],
            "eligible_roles": None,
            "provisional": 1 if provisional else 0,
            "insufficient": 1 if raw is None else 0,
            "updated_utc": stamp,
        }
        role_rows.append(row)
        if raw is not None and not provisional:
            by_role_valid[role].append(row)

    for role, rows in by_role_valid.items():
        if len(rows) < min_field_agents:
            for row in rows:
                row["provisional"] = 1
            continue
        vals = [float(r["raw_distance"]) for r in rows]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var)
        for row in rows:
            row["role_z"] = 0.0 if sd == 0 else (float(row["raw_distance"]) - mu) / sd

    overall_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in role_rows:
        if row["raw_distance"] is not None and row["role_z"] is not None and not row["provisional"]:
            overall_by_identity[row["identity_key"]].append(row)

    all_identities = {row["identity_key"]: row for row in role_rows}
    overall_rows: list[dict[str, Any]] = []
    for identity_key, sample in all_identities.items():
        rows = overall_by_identity.get(identity_key, [])
        weight_total = sum(min(int(r["valid_utterances"]), role_weight_cap) for r in rows)
        has_enough_roles = len(rows) >= min_eligible_roles
        if has_enough_roles and weight_total:
            overall_z = sum(float(r["role_z"]) * min(int(r["valid_utterances"]), role_weight_cap)
                            for r in rows) / weight_total
            raw_distance = sum(float(r["raw_distance"]) * min(int(r["valid_utterances"]), role_weight_cap)
                               for r in rows) / weight_total
        else:
            overall_z = None
            raw_distance = None
        owned = [r for r in role_rows if r["identity_key"] == identity_key]
        overall_rows.append({
            **{k: sample[k] for k in [
                "version", "identity_key", "agent_id", "display_name", "declared_model", "declared_harness",
            ]},
            "role": "*",
            "raw_distance": raw_distance,
            "role_z": None,
            "overall_z": overall_z,
            "valid_utterances": sum(int(r["valid_utterances"]) for r in owned),
            "valid_pairs": sum(int(r["valid_pairs"]) for r in owned),
            "sampled_pairs": sum(int(r["sampled_pairs"]) for r in owned),
            "excluded_pairs": sum(int(r["excluded_pairs"]) for r in owned),
            "eligible_roles": len(rows),
            "provisional": 1 if not has_enough_roles else 0,
            "insufficient": 1 if not has_enough_roles else 0,
            "updated_utc": stamp,
        })

    return overall_rows + role_rows


def placeholders(style: str, n: int) -> str:
    mark = "%s" if style == "pg" else "?"
    return ",".join([mark] * n)


def upsert_sql(table: str, cols: list[str], conflict: str, update_cols: list[str], style: str) -> str:
    update = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    return (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders(style, len(cols))}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {update}"
    )


def write_sqlite(path: Path, scores: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(SCORES_DDL)
        con.execute(JUDGMENTS_DDL)
        score_update = [c for c in SCORE_COLS if c not in {"version", "identity_key", "role"}]
        con.executemany(
            upsert_sql("creativity_scores", SCORE_COLS, "version,identity_key,role", score_update, "sqlite"),
            [tuple(row.get(c) for c in SCORE_COLS) for row in scores],
        )
        con.executemany(
            f"INSERT INTO creativity_judgments ({','.join(JUDGMENT_COLS)}) "
            f"VALUES ({placeholders('sqlite', len(JUDGMENT_COLS))}) "
            "ON CONFLICT (judgment_key) DO NOTHING",
            [tuple(row.get(c) for c in JUDGMENT_COLS) for row in judgments],
        )
        con.commit()
    finally:
        con.close()


def upload_pg(database_url: str, scores: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> None:
    import psycopg

    with psycopg.connect(database_url, connect_timeout=8) as con:
        with con.cursor() as cur:
            cur.execute("SET statement_timeout = 30000")
            cur.execute(SCORES_DDL)
            cur.execute(JUDGMENTS_DDL)
            score_update = [c for c in SCORE_COLS if c not in {"version", "identity_key", "role"}]
            cur.executemany(
                upsert_sql("creativity_scores", SCORE_COLS, "version,identity_key,role", score_update, "pg"),
                [tuple(row.get(c) for c in SCORE_COLS) for row in scores],
            )
            cur.executemany(
                f"INSERT INTO creativity_judgments ({','.join(JUDGMENT_COLS)}) "
                f"VALUES ({placeholders('pg', len(JUDGMENT_COLS))}) "
                "ON CONFLICT (judgment_key) DO NOTHING",
                [tuple(row.get(c) for c in JUDGMENT_COLS) for row in judgments],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the Persuasion Arena creativity signal")
    parser.add_argument("--sqlite", type=Path, default=Path("store/arena.db"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--version", default="creativity_v1")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Optional override. Omit to use the judge model/provider default.",
    )
    parser.add_argument("--max-pairs-per-bucket", type=int, default=12)
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--limit-buckets", type=int, default=0)
    parser.add_argument("--min-valid-utterances", type=int, default=4)
    parser.add_argument("--min-valid-pairs", type=int, default=4)
    parser.add_argument("--min-field-agents", type=int, default=3)
    parser.add_argument("--min-eligible-roles", type=int, default=2)
    parser.add_argument("--role-weight-cap", type=int, default=20)
    parser.add_argument("--write-sqlite", action="store_true")
    parser.add_argument("--upload-neon", action="store_true")
    args = parser.parse_args()

    env = load_env(args.env_file)
    utterances = load_utterances(args.sqlite)
    pairs, kept_buckets = select_pairs(
        utterances,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
        limit_buckets=args.limit_buckets or None,
    )
    print(f"utterances={len(utterances)} buckets={len(kept_buckets)} sampled_pairs={len(pairs)}")
    judgments = judge_pairs(
        pairs,
        env=env,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        batch_size=args.judge_batch_size,
        version=args.version,
    )
    scores = compute_scores(
        judgments,
        kept_buckets,
        version=args.version,
        min_valid_utterances=args.min_valid_utterances,
        min_valid_pairs=args.min_valid_pairs,
        min_field_agents=args.min_field_agents,
        min_eligible_roles=args.min_eligible_roles,
        role_weight_cap=args.role_weight_cap,
    )
    competitors = sum(1 for s in scores if s["role"] == "*")
    eligible = sum(1 for s in scores if s["role"] == "*" and not s["insufficient"])
    print(f"judgments={len(judgments)} score_rows={len(scores)} competitors={competitors} eligible={eligible}")

    if args.write_sqlite:
        write_sqlite(args.sqlite, scores, judgments)
        print(f"wrote sqlite snapshot version={args.version}")
    if args.upload_neon:
        database_url = env.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for --upload-neon")
        upload_pg(database_url, scores, judgments)
        print(f"uploaded Neon snapshot version={args.version}")


if __name__ == "__main__":
    main()
