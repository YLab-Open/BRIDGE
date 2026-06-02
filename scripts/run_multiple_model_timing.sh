#!/usr/bin/env bash
set -euo pipefail

# Start run_multiple_model.sh after a delay.
# Usage:
#   bash scripts/run_multiple_model_timing.sh          # default: wait 45 minutes
#   bash scripts/run_multiple_model_timing.sh 30       # wait 30 minutes

DELAY_MINUTES="${1:-45}"

if ! [[ "$DELAY_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid delay: ${DELAY_MINUTES}. Please provide minutes, for example: 45"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/log/multiple_model"

scheduled_at="$(date +"%m-%d_%H-%M-%S")"
schedule_log="${REPO_ROOT}/log/multiple_model/run_multiple_model_timing.${scheduled_at}.log"

nohup bash -c '
    set -euo pipefail
    delay_minutes="$1"
    script_dir="$2"

    echo "Scheduled at: $(date)"
    echo "Delay minutes: ${delay_minutes}"
    echo "Will start run_multiple_model.sh at approximately: $(date -d "+${delay_minutes} minutes" 2>/dev/null || true)"

    sleep "${delay_minutes}m"

    repo_root="$(cd "${script_dir}/.." && pwd)"
    echo "Starting run_multiple_model.sh at: $(date)"
    bash "${repo_root}/run_multiple_model.sh"
    echo "run_multiple_model.sh finished at: $(date)"
' _ "${DELAY_MINUTES}" "${SCRIPT_DIR}" > "${schedule_log}" 2>&1 &

echo "Delayed multi-model run scheduled."
echo "Delay: ${DELAY_MINUTES} minutes"
echo "Scheduler PID: $!"
echo "Scheduler log: ${schedule_log}"
echo "The actual model run logs will be created by run_multiple_model.sh."
