#!/usr/bin/env bash
# Connect YOUR agent — opencode pinned to GLM 5.2 — into a hosted connected run.
# Everything that kept getting dropped on copy-paste (PYTHONPATH, the model pin) is baked in here.
#
#   bash tools/play_glm.sh            # joins glm_demo_2 (the current hosted run)
#   bash tools/play_glm.sh <run_id>   # join a specific run
#   NAME='Vishal-GLM' bash tools/play_glm.sh   # override the displayed name
#
# Keep this terminal open until the games finish — don't Ctrl-C after "signed up".
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root, regardless of where you call it from

RUN="${1:-glm_demo_2}"
SERVER="${SERVER:-http://127.0.0.1:8021}"
NAME="${NAME:-GLM-opencode}"

export ARENA_OPENCODE_MODEL="openrouter/z-ai/glm-5.2"   # the GLM 5.2 pin (fixes opencode's no-model hang)
export PYTHONPATH=.                                      # so the harness can import `examples`

echo "joining run=$RUN as '$NAME' (opencode @ $ARENA_OPENCODE_MODEL) -> $SERVER"
exec .venv/bin/arena-agent play \
  --run "$RUN" --server "$SERVER" --name "$NAME" examples/opencode_agent.py
