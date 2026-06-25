"""Evaluate WolfForgeAgentV2 against a frozen baseline over saved connected runs.

The value of V2 is whether it MEASURABLY beats a frozen Charisma baseline with model, temperature,
token caps, reasoning effort, opponents, seeds, seats, and game settings held constant. This tool
reads finished runs from the active store, pairs the V2 seat against the baseline seat WITHIN each
game (the game is the unit of uncertainty), and emits a CSV of game-level observations, a JSON
summary, and a Markdown report.

Statistics are game-clustered on purpose: an exact paired McNemar test on per-game win/loss pairs,
and a bootstrap CI that resamples WHOLE GAMES. No naive independent-seat p-value is reported as the
main result, because both agents play the same correlated games.

Pure analysis functions (analyze / mcnemar_exact / bootstrap_diff_ci / manifest_warnings) take plain
data so they can be unit-tested on deterministic fixtures with no store and no network
(see tests/test_wolfforge_v2_eval.py).

Usage:
    python tools/wolfforge_v2_eval.py --runs holdout_a holdout_b \
        --v2-name WolfForgeV2 --baseline-name CharismaBaseline \
        --out-dir ./wf_v2_report [--telemetry v2_telemetry.jsonl] [--manifest frozen_v2.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------------------

@dataclass
class GameObs:
    """One game's paired observation: the V2 seat vs the baseline seat in the same game."""
    run_id: str
    gid: int
    v2_won: bool
    baseline_won: bool
    v2_role: str
    baseline_role: str
    v2_team: str
    baseline_team: str
    v2_forfeits: int = 0
    v2_calls: int = 0
    baseline_forfeits: int = 0
    baseline_calls: int = 0


# --------------------------------------------------------------------------------------------------
# Loading from the store (impure boundary kept thin)
# --------------------------------------------------------------------------------------------------

def load_observations(run_ids: list[str], v2_name: str, baseline_name: str,
                      store_module: Any) -> list[GameObs]:
    """Read player rows for each run and pair the V2 seat with the baseline seat per game.

    A game contributes a paired observation only if BOTH a V2-named seat and a baseline-named seat
    are present (the primary comparison requires them head to head in the same game). When multiple
    rows share a name in a game, the lowest-seat row is used.
    """
    obs: list[GameObs] = []
    for run_id in run_ids:
        rows = store_module.player_rows(run_id)
        by_game: dict[int, list[dict]] = {}
        for r in rows:
            by_game.setdefault(r["gid"], []).append(r)
        for gid, seats in sorted(by_game.items()):
            v2 = _pick(seats, v2_name)
            base = _pick(seats, baseline_name)
            if not v2 or not base:
                continue
            obs.append(GameObs(
                run_id=run_id, gid=gid,
                v2_won=bool(v2["won"]), baseline_won=bool(base["won"]),
                v2_role=v2.get("dealt_role") or "?", baseline_role=base.get("dealt_role") or "?",
                v2_team=v2.get("team") or "?", baseline_team=base.get("team") or "?",
                v2_forfeits=int(v2.get("forfeits") or 0), v2_calls=int(v2.get("calls") or 0),
                baseline_forfeits=int(base.get("forfeits") or 0),
                baseline_calls=int(base.get("calls") or 0),
            ))
    return obs


def _pick(seats: list[dict], name: str) -> dict | None:
    matches = sorted((s for s in seats if s.get("agent") == name), key=lambda s: s.get("seat", 0))
    return matches[0] if matches else None


# --------------------------------------------------------------------------------------------------
# Statistics (stdlib only — the game is the cluster)
# --------------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(lo, hi, point) for k successes in n trials. Matches arena.score.wilson."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half), p)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant pair counts.

    b = games V2 won and baseline lost; c = games V2 lost and baseline won. Under H0 each discordant
    pair is a fair coin, so we compute the exact two-sided binomial tail probability. Returns 1.0
    when there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_diff_ci(obs: list[GameObs], iterations: int = 10000, seed: int = 12345,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for (V2 win rate - baseline win rate), resampling WHOLE GAMES.

    The game is the resampling unit because both agents share the same correlated games. Seeded for
    reproducibility.
    """
    n = len(obs)
    if n == 0:
        return (0.0, 0.0)
    diffs = [(1 if o.v2_won else 0) - (1 if o.baseline_won else 0) for o in obs]
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        stats.append(total / n)
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * iterations))]
    hi = stats[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


# --------------------------------------------------------------------------------------------------
# Core analysis (pure)
# --------------------------------------------------------------------------------------------------

@dataclass
class Analysis:
    n_games: int
    v2_wins: int
    baseline_wins: int
    v2_rate: float
    baseline_rate: float
    effect_pp: float
    ci_low_pp: float
    ci_high_pp: float
    mcnemar_p: float
    discordant_b: int  # V2 won, baseline lost
    discordant_c: int  # V2 lost, baseline won
    v2_forfeit_rate: float
    baseline_forfeit_rate: float
    by_role: dict[str, dict] = field(default_factory=dict)
    by_team: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_games": self.n_games,
            "v2_wins": self.v2_wins, "baseline_wins": self.baseline_wins,
            "v2_rate": round(self.v2_rate, 4), "baseline_rate": round(self.baseline_rate, 4),
            "effect_pp": round(self.effect_pp, 2),
            "ci95_pp": [round(self.ci_low_pp, 2), round(self.ci_high_pp, 2)],
            "mcnemar_p": round(self.mcnemar_p, 5),
            "discordant": {"v2_only": self.discordant_b, "baseline_only": self.discordant_c},
            "v2_forfeit_rate": round(self.v2_forfeit_rate, 4),
            "baseline_forfeit_rate": round(self.baseline_forfeit_rate, 4),
            "by_role": self.by_role,
            "by_team": self.by_team,
        }


def analyze(obs: list[GameObs], *, bootstrap_iterations: int = 10000,
            bootstrap_seed: int = 12345) -> Analysis:
    n = len(obs)
    v2_wins = sum(1 for o in obs if o.v2_won)
    base_wins = sum(1 for o in obs if o.baseline_won)
    v2_rate = v2_wins / n if n else 0.0
    base_rate = base_wins / n if n else 0.0
    b = sum(1 for o in obs if o.v2_won and not o.baseline_won)
    c = sum(1 for o in obs if (not o.v2_won) and o.baseline_won)
    lo, hi = bootstrap_diff_ci(obs, iterations=bootstrap_iterations, seed=bootstrap_seed)

    v2_calls = sum(o.v2_calls for o in obs)
    v2_ff = sum(o.v2_forfeits for o in obs)
    base_calls = sum(o.baseline_calls for o in obs)
    base_ff = sum(o.baseline_forfeits for o in obs)

    by_role: dict[str, dict] = {}
    roles = sorted({o.v2_role for o in obs})
    for role in roles:
        sub = [o for o in obs if o.v2_role == role]
        w = sum(1 for o in sub if o.v2_won)
        rlo, rhi, rp = wilson(w, len(sub))
        by_role[role] = {"games": len(sub), "v2_wins": w, "v2_rate": round(rp, 3),
                         "ci95": [round(rlo, 3), round(rhi, 3)]}

    by_team: dict[str, dict] = {}
    for team in sorted({o.v2_team for o in obs}):
        sub = [o for o in obs if o.v2_team == team]
        w = sum(1 for o in sub if o.v2_won)
        tlo, thi, tp = wilson(w, len(sub))
        by_team[team] = {"games": len(sub), "v2_wins": w, "v2_rate": round(tp, 3),
                         "ci95": [round(tlo, 3), round(thi, 3)]}

    return Analysis(
        n_games=n, v2_wins=v2_wins, baseline_wins=base_wins,
        v2_rate=v2_rate, baseline_rate=base_rate,
        effect_pp=(v2_rate - base_rate) * 100,
        ci_low_pp=lo * 100, ci_high_pp=hi * 100,
        mcnemar_p=mcnemar_exact(b, c), discordant_b=b, discordant_c=c,
        v2_forfeit_rate=(v2_ff / v2_calls) if v2_calls else 0.0,
        baseline_forfeit_rate=(base_ff / base_calls) if base_calls else 0.0,
        by_role=by_role, by_team=by_team,
    )


# --------------------------------------------------------------------------------------------------
# Configuration manifest + mismatch warnings
# --------------------------------------------------------------------------------------------------

_MANIFEST_FIELDS = ("model", "temperature", "reasoning_effort", "max_tokens",
                    "deck_preset", "discussion_rounds")


def run_manifest(run_id: str, store_module: Any) -> dict:
    """Extract the comparable configuration of a run from stored metadata."""
    run = store_module.get_run(run_id) or {}
    agents = run.get("agents") or []
    cfg = (run.get("metadata") or {}).get("run_config") or {}
    models = sorted({a.get("model") for a in agents if a.get("model")})
    temps = sorted({a.get("temperature") for a in agents if a.get("temperature") is not None})
    efforts = sorted({a.get("reasoning_effort") for a in agents if a.get("reasoning_effort")})
    maxtoks = sorted({a.get("max_tokens") for a in agents if a.get("max_tokens") is not None})
    return {
        "run_id": run_id,
        "model": models or ([cfg["model"]] if cfg.get("model") else []),
        "temperature": temps or ([cfg["temperature"]] if cfg.get("temperature") is not None else []),
        "reasoning_effort": efforts or ([cfg["reasoning_effort"]] if cfg.get("reasoning_effort") else []),
        "max_tokens": maxtoks or ([cfg["max_tokens_per_turn"]] if cfg.get("max_tokens_per_turn") is not None else []),
        "deck_preset": [run.get("deck_preset")] if run.get("deck_preset") else [],
        "discussion_rounds": [cfg["discussion_rounds"]] if cfg.get("discussion_rounds") is not None else [],
        "seed_base": run.get("seed_base"),
        "n_games": run.get("n_games"),
    }


def manifest_warnings(manifests: list[dict], frozen: dict | None = None) -> list[str]:
    """Return prominent warnings when compared runs differ in any setting that invalidates the
    comparison, or differ from the frozen V2 manifest. An empty list means the comparison is clean."""
    warnings: list[str] = []
    for fld in _MANIFEST_FIELDS:
        values = set()
        for m in manifests:
            for v in m.get(fld, []):
                values.add(v)
        if len(values) > 1:
            warnings.append(f"MISMATCH: runs disagree on '{fld}': {sorted(map(str, values))}. "
                            f"The V2-vs-baseline comparison is NOT controlled for {fld}.")
    if frozen:
        for fld in _MANIFEST_FIELDS:
            frozen_val = frozen.get(fld)
            if frozen_val is None:
                continue
            observed = set()
            for m in manifests:
                observed.update(m.get(fld, []))
            # Warn unless every run used exactly the frozen value for this field.
            if observed and observed != {frozen_val}:
                warnings.append(f"MISMATCH vs frozen manifest: '{fld}' frozen={frozen_val!r} "
                                f"but runs used {sorted(map(str, observed))}.")
    return warnings


# --------------------------------------------------------------------------------------------------
# Telemetry summary (optional, heuristic)
# --------------------------------------------------------------------------------------------------

def summarize_telemetry(path: str | None, agent_name: str | None = None) -> dict | None:
    """Aggregate a JSONL telemetry log into latency/token/repair/fallback summaries. Optional; the
    primary win-rate result never depends on it."""
    if not path or not os.path.exists(path):
        return None
    lat: list[int] = []
    pt: list[int] = []
    ct: list[int] = []
    repairs = fallbacks = invalid = total = 0
    memory_modes: set = set()
    memory_hashes: set = set()
    memory_budget = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if agent_name and rec.get("agent_name") != agent_name:
                continue
            total += 1
            if isinstance(rec.get("latency_ms"), int):
                lat.append(rec["latency_ms"])
            if isinstance(rec.get("prompt_tokens"), int):
                pt.append(rec["prompt_tokens"])
            if isinstance(rec.get("completion_tokens"), int):
                ct.append(rec["completion_tokens"])
            repairs += 1 if rec.get("repair_attempted") else 0
            fallbacks += 1 if rec.get("fallback_used") else 0
            invalid += 0 if rec.get("action_valid") else 1
            if rec.get("memory_mode"):
                memory_modes.add(rec["memory_mode"])
            if rec.get("memory_snapshot_hash"):
                memory_hashes.add(rec["memory_snapshot_hash"])
            if isinstance(rec.get("memory_context_chars"), int):
                memory_budget = max(memory_budget, rec["memory_context_chars"])
    return {
        "turns": total,
        "median_latency_ms": _percentile(lat, 50),
        "p95_latency_ms": _percentile(lat, 95),
        "prompt_tokens_total": sum(pt),
        "completion_tokens_total": sum(ct),
        "repair_rate": round(repairs / total, 4) if total else 0.0,
        "fallback_rate": round(fallbacks / total, 4) if total else 0.0,
        "invalid_rate": round(invalid / total, 4) if total else 0.0,
        # Cross-run memory metadata (used for mismatch warnings; never the memory contents).
        "memory_modes": sorted(memory_modes),
        "memory_snapshot_hashes": sorted(memory_hashes),
        "memory_max_context_chars": memory_budget,
    }


def memory_warnings(telemetry: dict | None) -> list[str]:
    """Warn when a single compared telemetry set mixes memory modes/snapshots — that is a DIFFERENT
    experiment from a clean prompt-policy comparison (see WOLFFORGE_V2_EVAL.md)."""
    if not telemetry:
        return []
    out: list[str] = []
    modes = telemetry.get("memory_modes") or []
    hashes = telemetry.get("memory_snapshot_hashes") or []
    if len(modes) > 1:
        out.append(f"MEMORY MISMATCH: compared runs mix memory modes {modes}. A memory-on agent is "
                   f"NOT a clean prompt-policy comparison against a frozen baseline.")
    if len(hashes) > 1:
        out.append(f"MEMORY MISMATCH: compared runs used different memory snapshots {hashes}. Freeze "
                   f"one snapshot for a holdout.")
    if any(m in ("write", "readwrite") for m in modes):
        out.append("MEMORY WARNING: a write/readwrite memory mode appears in the compared runs; memory "
                   "was mutating during evaluation (development mode, not a frozen holdout).")
    return out


def _percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int((pct / 100) * len(s)))
    return s[idx]


# --------------------------------------------------------------------------------------------------
# Output renderers
# --------------------------------------------------------------------------------------------------

def write_csv(obs: list[GameObs], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "gid", "v2_won", "baseline_won", "v2_role", "baseline_role",
                    "v2_team", "baseline_team", "v2_forfeits", "v2_calls",
                    "baseline_forfeits", "baseline_calls"])
        for o in obs:
            w.writerow([o.run_id, o.gid, int(o.v2_won), int(o.baseline_won), o.v2_role,
                        o.baseline_role, o.v2_team, o.baseline_team, o.v2_forfeits, o.v2_calls,
                        o.baseline_forfeits, o.baseline_calls])


def render_markdown(analysis: Analysis, manifests: list[dict], warnings: list[str],
                    v2_name: str, baseline_name: str, telemetry: dict | None,
                    success_threshold_pp: float = 5.0) -> str:
    a = analysis
    lines: list[str] = ["# WolfForgeAgentV2 evaluation report", ""]

    if warnings:
        lines += ["> ⚠️ **CONFIGURATION MISMATCH — results are NOT a controlled comparison:**", ""]
        lines += [f"> - {w}" for w in warnings]
        lines.append("")
    else:
        lines += ["> ✅ Compared runs share model, temperature, token caps, reasoning effort, deck, "
                  "and discussion rounds.", ""]

    lines += ["## Configuration manifest", ""]
    for m in manifests:
        lines.append(f"- `{m['run_id']}`: model={m['model']} temp={m['temperature']} "
                     f"effort={m['reasoning_effort']} max_tokens={m['max_tokens']} "
                     f"deck={m['deck_preset']} rounds={m['discussion_rounds']} "
                     f"seed_base={m['seed_base']} n_games={m['n_games']}")
    lines.append("")

    lines += [
        "## Primary result", "",
        f"Paired games (both `{v2_name}` and `{baseline_name}` seated): **{a.n_games}**", "",
        f"- {v2_name}: {a.v2_wins}/{a.n_games} = **{a.v2_rate*100:.1f}%**",
        f"- {baseline_name}: {a.baseline_wins}/{a.n_games} = **{a.baseline_rate*100:.1f}%**",
        f"- Effect (V2 − baseline): **{a.effect_pp:+.1f} pp**", "",
        "## 95% interval (whole-game bootstrap)", "",
        f"Difference in win rate, resampling whole games: **[{a.ci_low_pp:+.1f}, {a.ci_high_pp:+.1f}] pp**",
        "",
        "## Paired test", "",
        f"Exact McNemar (game-clustered): discordant pairs b={a.discordant_b} (V2-only wins), "
        f"c={a.discordant_c} (baseline-only wins); **p = {a.mcnemar_p:.4f}**.",
        "",
        "## Reliability (forfeits / repairs / fallbacks)", "",
        f"- {v2_name} forfeit rate: {a.v2_forfeit_rate*100:.2f}%",
        f"- {baseline_name} forfeit rate: {a.baseline_forfeit_rate*100:.2f}%",
    ]
    if telemetry:
        lines += [
            f"- V2 repair rate: {telemetry['repair_rate']*100:.2f}% · fallback rate: "
            f"{telemetry['fallback_rate']*100:.2f}% · invalid rate: {telemetry['invalid_rate']*100:.2f}%",
        ]
    else:
        lines.append("- Repair/fallback/invalid rates: not available (no --telemetry log supplied).")
    lines.append("")

    lines += ["## Latency and token use", ""]
    if telemetry:
        lines += [
            f"- Turns logged: {telemetry['turns']}",
            f"- Latency median / p95: {telemetry['median_latency_ms']} ms / {telemetry['p95_latency_ms']} ms",
            f"- Prompt / completion tokens (total): {telemetry['prompt_tokens_total']} / "
            f"{telemetry['completion_tokens_total']}",
        ]
    else:
        lines.append("- Not available (no --telemetry log supplied).")
    lines.append("")

    lines += ["## Cross-run memory", ""]
    if telemetry and (telemetry.get("memory_modes") or telemetry.get("memory_snapshot_hashes")):
        lines += [
            f"- modes seen: {telemetry.get('memory_modes')}",
            f"- snapshot hashes: {telemetry.get('memory_snapshot_hashes')}",
            f"- max injected memory chars: {telemetry.get('memory_max_context_chars')}",
        ]
    else:
        lines.append("- memory off (or no telemetry) — clean default-V2 comparison.")
    lines.append("")

    lines += ["## Role / objective breakdown (V2 dealt role)", "",
              "| Role | Games | V2 wins | V2 win% | 95% CI |", "|---|---:|---:|---:|---:|"]
    for role, d in a.by_role.items():
        lines.append(f"| {role} | {d['games']} | {d['v2_wins']} | {d['v2_rate']*100:.0f}% | "
                     f"{d['ci95'][0]*100:.0f}–{d['ci95'][1]*100:.0f}% |")
    lines += ["", "### By team (V2 dealt team)", "", "| Team | Games | V2 wins | V2 win% |",
              "|---|---:|---:|---:|"]
    for team, d in a.by_team.items():
        lines.append(f"| {team} | {d['games']} | {d['v2_wins']} | {d['v2_rate']*100:.0f}% |")
    lines.append("")

    lines += [
        "## Limitations", "",
        "- Mixed-table evaluation measures relative tournament performance with strategic "
        "interference between agents; it is not a pure 1v1 skill measurement.",
        "- Behavioral counters derived from telemetry are heuristic, not ground truth.",
        "- Role-specific cells are small; treat per-role rates as directional.",
        "- If the mismatch banner is present above, treat the headline number as invalid.",
        "",
        "## Honest conclusion", "",
    ]
    improved = (a.effect_pp >= success_threshold_pp and a.ci_high_pp > 0
                and a.v2_forfeit_rate <= a.baseline_forfeit_rate + 1e-9)
    if warnings:
        lines.append("Comparison is not controlled (see banner); no conclusion is drawn.")
    elif improved and a.ci_low_pp > 0:
        lines.append(f"V2 improves on the frozen baseline by {a.effect_pp:+.1f} pp with a 95% "
                     f"interval excluding zero and no worse forfeit rate. This meets the pre-"
                     f"registered success criterion (≥ {success_threshold_pp:.0f} pp, directionally "
                     f"favorable interval, forfeits no worse).")
    elif improved:
        lines.append(f"V2 is directionally better ({a.effect_pp:+.1f} pp) and meets the point "
                     f"threshold, but the 95% interval includes zero — not yet conclusive.")
    else:
        lines.append(f"V2 does not clear the success bar: effect {a.effect_pp:+.1f} pp "
                     f"(95% CI [{a.ci_low_pp:+.1f}, {a.ci_high_pp:+.1f}] pp), forfeit rate "
                     f"{a.v2_forfeit_rate*100:.2f}% vs {a.baseline_forfeit_rate*100:.2f}%.")
    lines.append("")
    return "\n".join(lines)


def build_report(run_ids: list[str], v2_name: str, baseline_name: str, store_module: Any,
                 *, telemetry_path: str | None = None, frozen_manifest: dict | None = None,
                 bootstrap_iterations: int = 10000) -> dict:
    """End-to-end analysis bundle (no file writes). Returns observations, analysis, manifests,
    warnings, telemetry, and the rendered markdown."""
    obs = load_observations(run_ids, v2_name, baseline_name, store_module)
    analysis = analyze(obs, bootstrap_iterations=bootstrap_iterations)
    manifests = [run_manifest(r, store_module) for r in run_ids]
    telemetry = summarize_telemetry(telemetry_path, v2_name)
    warnings = manifest_warnings(manifests, frozen_manifest) + memory_warnings(telemetry)
    markdown = render_markdown(analysis, manifests, warnings, v2_name, baseline_name, telemetry)
    return {"observations": obs, "analysis": analysis, "manifests": manifests,
            "warnings": warnings, "telemetry": telemetry, "markdown": markdown}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate WolfForgeAgentV2 vs a frozen baseline")
    p.add_argument("--runs", nargs="+", required=True, help="run ids to analyze (holdout blocks)")
    p.add_argument("--v2-name", default="WolfForgeV2")
    p.add_argument("--baseline-name", default="CharismaBaseline")
    p.add_argument("--out-dir", default="./wolfforge_v2_report")
    p.add_argument("--telemetry", default=None, help="optional V2 telemetry JSONL for latency/cost")
    p.add_argument("--manifest", default=None, help="optional frozen V2 manifest JSON to cross-check")
    p.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = p.parse_args(argv)

    from arena import store  # imported lazily so unit tests need no store

    frozen = None
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, encoding="utf-8") as f:
            frozen = json.load(f)

    report = build_report(args.runs, args.v2_name, args.baseline_name, store,
                          telemetry_path=args.telemetry, frozen_manifest=frozen,
                          bootstrap_iterations=args.bootstrap_iterations)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(report["observations"], os.path.join(args.out_dir, "games.csv"))
    summary = {
        "v2_name": args.v2_name, "baseline_name": args.baseline_name, "runs": args.runs,
        "analysis": report["analysis"].to_dict(), "manifests": report["manifests"],
        "warnings": report["warnings"], "telemetry": report["telemetry"],
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report["markdown"])

    for w in report["warnings"]:
        print(f"[wolfforge-v2-eval] {w}", flush=True)
    a = report["analysis"]
    print(f"[wolfforge-v2-eval] {args.v2_name} {a.v2_rate*100:.1f}% vs {args.baseline_name} "
          f"{a.baseline_rate*100:.1f}% ({a.effect_pp:+.1f} pp, McNemar p={a.mcnemar_p:.4f}, "
          f"n={a.n_games}) -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
