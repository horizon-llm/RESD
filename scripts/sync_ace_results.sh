#!/usr/bin/env bash
#
# Periodically upload newly created or modified checkpoint files to S3.
# One-way only: local -> S3.
#
# See original header for usage. Added features:
#   --verbose              Show per-file upload lines (removes --only-show-errors).
#   --uploaded-log <file>  Append lines of uploaded objects only.
#   --manifest-each        After each sync, write a manifest of S3 objects.
#   LOG_FILE=<path>        (env) If set, append all script logs there (still prints to stdout if tailing).
#   --rate-limit <rate>    Cap S3 bandwidth (e.g., 50MB/s, 300Kb/s). Writes to AWS config for profile.
#   --concurrency <n>      Set S3 max concurrent requests (e.g., 5, 10). Writes to AWS config for profile.
#
set -euo pipefail

DEFAULT_S3="s3://shopqa-users/yuwzhan/ace/results"

INTERVAL="${INTERVAL:-600}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./results}"
S3_URI="${S3_URI:-$DEFAULT_S3}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT=0
RUN_ONCE=0
QUIET=0
VERBOSE=0
UPLOADED_LOG=""
MANIFEST_EACH=0
RATE_LIMIT="100MB/s"
MAX_CONCURRENT="5"

EXCLUDES=()
EXTRAS=()

print_help() {
  cat <<'EOF'
Usage: sync_checkpoints_loop.sh [options]

Core:
  --interval <seconds>     Frequency (default 600 or INTERVAL env)
  --dir <path>             Local checkpoints dir (default ./checkpoints)
  --s3 <uri>               Destination S3 URI
  --once                   Run a single sync then exit
  --timeout <seconds>      Per-sync timeout (needs coreutils timeout)
  --exclude <pattern>      Exclude pattern (repeatable)
  --extra <arg>            Extra aws s3 sync arg (repeatable)

Rate limiting:
  --rate-limit <rate>      Cap S3 bandwidth (e.g., 50MB/s, 300Kb/s)
  --concurrency <n>        Max concurrent S3 requests (default CLI is 10)

Logging / Auditing:
  --verbose                Show per-file upload lines
  --uploaded-log <file>    Append 'upload:' lines only to this file
  --manifest-each          Generate a manifest file after each sync
  -q, --quiet              Suppress informational logs (errors still shown)
  DRY_RUN=1                Preview; no uploads
  LOG_FILE=<file>          Append script logs to a file

Other:
  -h, --help               This help

Examples:
  ./sync_checkpoints_loop.sh --interval 300 --verbose
  LOG_FILE=logs/ckpt_sync.log ./sync_checkpoints_loop.sh --uploaded-log logs/uploaded_files.log
  ./sync_checkpoints_loop.sh --once --verbose
  ./sync_checkpoints_loop.sh --rate-limit 50MB/s --concurrency 5
EOF
  exit 0
}

log() {
  local ts
  if [ -n "${LOG_TS_FORMAT:-}" ]; then
    ts="$(date +"$LOG_TS_FORMAT")"
  else
    ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  fi
  local line="[$ts] $*"
  [ "$QUIET" != "1" ] && echo "$line"
  if [ -n "${LOG_FILE:-}" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "$line" >> "$LOG_FILE"
  fi
}

err() { >&2 log "ERROR: $*"; }

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2;;
    --dir) CHECKPOINTS_DIR="$2"; shift 2;;
    --s3) S3_URI="$2"; shift 2;;
    --exclude) EXCLUDES+=("$2"); shift 2;;
    --extra) EXTRAS+=("$2"); shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --once) RUN_ONCE=1; shift;;
    -q|--quiet) QUIET=1; shift;;
    --verbose) VERBOSE=1; shift;;
    --uploaded-log) UPLOADED_LOG="$2"; shift 2;;
    --manifest-each) MANIFEST_EACH=1; shift;;
    --rate-limit) RATE_LIMIT="$2"; shift 2;;
    --concurrency) MAX_CONCURRENT="$2"; shift 2;;
    -h|--help) print_help;;
    *) err "Unknown arg: $1"; exit 1;;
  esac
done

if ! command -v aws >/dev/null 2>&1; then
  err "aws CLI not found."
  exit 1
fi

# Apply optional AWS CLI S3 rate/concurrency limits to the active profile.
# This modifies ~/.aws/config for $AWS_DEFAULT_PROFILE on purpose.
if [ -n "$RATE_LIMIT" ]; then
  log "Applying S3 bandwidth cap: $RATE_LIMIT (profile: ${AWS_DEFAULT_PROFILE:-default})"
  if ! aws configure set s3.max_bandwidth "$RATE_LIMIT" --profile "${AWS_DEFAULT_PROFILE:-default}"; then
    err "Failed to set s3.max_bandwidth"
  fi
  # Ensure limit is honored by forcing classic transfer client (CRT can ignore max_bandwidth).
  aws configure set s3.preferred_transfer_client classic --profile "${AWS_DEFAULT_PROFILE:-default}" || true
fi
if [ -n "$MAX_CONCURRENT" ]; then
  log "Setting S3 max concurrent requests: $MAX_CONCURRENT (profile: ${AWS_DEFAULT_PROFILE:-default})"
  if ! aws configure set s3.max_concurrent_requests "$MAX_CONCURRENT" --profile "${AWS_DEFAULT_PROFILE:-default}"; then
    err "Failed to set s3.max_concurrent_requests"
  fi
fi

mkdir -p "$CHECKPOINTS_DIR"

# Build base command
SYNC_BASE=(aws s3 sync "$CHECKPOINTS_DIR" "$S3_URI" --no-progress)
if [ "$VERBOSE" -eq 0 ]; then
  SYNC_BASE+=(--only-show-errors)
fi
for pat in "${EXCLUDES[@]}"; do
  SYNC_BASE+=(--exclude "$pat")
done
[ "$DRY_RUN" = "1" ] && SYNC_BASE+=(--dryrun)
[ "${#EXTRAS[@]}" -gt 0 ] && SYNC_BASE+=("${EXTRAS[@]}")

log "Starting checkpoint uploader"
log "Dir: $CHECKPOINTS_DIR -> $S3_URI | Interval: ${INTERVAL}s | Verbose: $VERBOSE | Once: $RUN_ONCE"
[ -n "$RATE_LIMIT" ] && log "Rate limit: $RATE_LIMIT"
[ -n "$MAX_CONCURRENT" ] && log "Max concurrent requests: $MAX_CONCURRENT"
[ "$DRY_RUN" = "1" ] && log "DRY RUN active"
[ "$MANIFEST_EACH" = "1" ] && log "Manifest generation enabled"

# Concurrency guard
LOCK_FILE="${CHECKPOINTS_DIR%/}/.sync_checkpoints.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  err "Another instance is running (lock: $LOCK_FILE). Exiting."
  exit 1
fi

stop=0
trap 'stop=1; log "Signal received, will exit after current cycle."' INT TERM

generate_manifest() {
  local manifest_dir="${CHECKPOINTS_DIR%/}/.manifests"
  mkdir -p "$manifest_dir"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local manifest_file="${manifest_dir}/manifest_${stamp}.txt"
  # List objects (size date path)
  if aws s3 ls "$S3_URI" --recursive > "$manifest_file"; then
    log "Manifest written: $manifest_file"
    # Optional: also upload manifest to S3 (commented)
    # aws s3 cp "$manifest_file" "${S3_URI%/}/manifests/"
  else
    err "Failed to generate manifest."
  fi
}

do_sync() {
  log "Sync start"
  local output tmpfile rc=0
  tmpfile="$(mktemp)"
  # Run command with or without timeout
  if [ "$TIMEOUT" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "$TIMEOUT" "${SYNC_BASE[@]}" &> "$tmpfile"
    rc=$?
    set -e
  else
    set +e
    "${SYNC_BASE[@]}" &> "$tmpfile"
    rc=$?
    set -e
  fi

  # Print output if verbose (or if errors occurred)
  if [ "$VERBOSE" -eq 1 ] || [ $rc -ne 0 ]; then
    while IFS= read -r line; do
      # Filter progress lines (shouldn't appear with --no-progress)
      echo "$line"
      [ -n "${LOG_FILE:-}" ] && echo "$line" >> "$LOG_FILE"
    done <"$tmpfile"
  fi

  # Extract uploaded lines to uploaded log
  if [ -n "$UPLOADED_LOG" ]; then
    mkdir -p "$(dirname "$UPLOADED_LOG")"
    grep -E '^upload:' "$tmpfile" >> "$UPLOADED_LOG" || true
  fi

  if [ $rc -ne 0 ]; then
    err "Sync command failed (exit $rc)"
    rm -f "$tmpfile"
    return $rc
  fi

  rm -f "$tmpfile"
  log "Sync done"

  if [ "$MANIFEST_EACH" -eq 1 ]; then
    generate_manifest
  fi
}

# Initial sync
do_sync || err "Initial sync encountered an error."

if [ "$RUN_ONCE" -eq 1 ]; then
  log "Completed single sync; exiting."
  exit 0
fi

while [ $stop -eq 0 ]; do
  sleep "$INTERVAL" || true
  [ $stop -ne 0 ] && break
  do_sync || err "Periodic sync encountered an error."
done

log "Exited cleanly."
