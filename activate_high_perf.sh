#!/usr/bin/env bash
#
# Source this wrapper to apply the high-performance TensorFlow configuration.
# It exists primarily so other tooling can rely on a stable filename.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1090
source "${SCRIPT_DIR}/high_perf_env.sh"

return 0 2>/dev/null || exit 0
