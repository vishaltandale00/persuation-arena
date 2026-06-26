#!/usr/bin/env bash
#
# tools/setup_spawn_secret.sh — provision the shared secret that lets the Vercel
# central API trigger the Modal per-run coordinator spawn endpoint.
#
# The same token must live in BOTH places (it's compared with hmac.compare_digest),
# so this generates it once and upserts it to Modal + Vercel together.
#
# Idempotent: re-running regenerates a NEW token (rotates) unless you pin one via
#   SPAWN_TOKEN=pa_spawn_xxx ./tools/setup_spawn_secret.sh
#
# Two phases — the URL only exists after the endpoint is deployed:
#   Phase 1 (now):                 ./tools/setup_spawn_secret.sh
#   Phase 2 (after `modal deploy`): ./tools/setup_spawn_secret.sh --url https://<workspace>--spawn.modal.run
#
# Env overrides: MODAL (default .venv/bin/modal), VERCEL (default vercel),
#                VERCEL_ENV (default production), SPAWN_TOKEN (reuse an existing token).
set -euo pipefail

MODAL="${MODAL:-.venv/bin/modal}"
VERCEL="${VERCEL:-vercel}"
VERCEL_ENV="${VERCEL_ENV:-production}"
SECRET_NAME="arena-spawn-token"

# Upsert a Vercel env var (rm is best-effort; add reads stdin with NO trailing newline,
# which matters — a trailing \n would break the token match).
vercel_upsert() {  # $1=name $2=value
  "$VERCEL" env rm "$1" "$VERCEL_ENV" -y >/dev/null 2>&1 || true
  printf %s "$2" | "$VERCEL" env add "$1" "$VERCEL_ENV" >/dev/null
  echo "  vercel: $1 set for $VERCEL_ENV"
}

# ---- Phase 2: wire the URL after deploy ------------------------------------
if [[ "${1:-}" == "--url" ]]; then
  URL="${2:?usage: $0 --url https://<workspace>--spawn.modal.run}"
  command -v "$VERCEL" >/dev/null 2>&1 || { echo "vercel CLI not found (npm i -g vercel)"; exit 1; }
  vercel_upsert ARENA_SPAWN_URL "$URL"
  echo "✓ ARENA_SPAWN_URL wired. Redeploy Vercel (git push to main, or \`vercel --prod\`) to pick it up."
  exit 0
fi

# ---- Phase 1: generate token + set on Modal and Vercel ---------------------
command -v "$MODAL"  >/dev/null 2>&1 || { echo "modal CLI not found at '$MODAL' (set \$MODAL, or run: $MODAL token new)"; exit 1; }
command -v "$VERCEL" >/dev/null 2>&1 || { echo "vercel CLI not found (npm i -g vercel)"; exit 1; }

TOKEN="${SPAWN_TOKEN:-pa_spawn_$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"

# Modal secret. --force overwrites an existing secret so re-runs are clean.
# (If your modal version rejects --force, drop it — first creation works without it.)
"$MODAL" secret create "$SECRET_NAME" "ARENA_SPAWN_TOKEN=$TOKEN" --force >/dev/null
echo "  modal:  secret '$SECRET_NAME' upserted (key ARENA_SPAWN_TOKEN)"

vercel_upsert ARENA_SPAWN_TOKEN "$TOKEN"

echo
echo "✓ Spawn token provisioned on Modal + Vercel ($VERCEL_ENV)."
echo "  Save this token in your vault — re-run with SPAWN_TOKEN=... to reuse it later:"
echo "    $TOKEN"
echo
echo "Next: once the spawn endpoint is added and you run"
echo "    $MODAL deploy arena/modal_app.py"
echo "copy the printed spawn URL and run:"
echo "    $0 --url https://<workspace>--spawn.modal.run"
