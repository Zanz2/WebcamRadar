# Shared preamble: run from the repository root no matter where the script is called from.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source calibration/settings.sh
cal() { uv run p2v-calibrate "$@"; }
need() { [ -e "$1" ] || { echo "Missing $1 -- $2" >&2; exit 1; }; }
