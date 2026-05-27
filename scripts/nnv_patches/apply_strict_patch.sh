#!/usr/bin/env bash
# Thin shell wrapper around apply_strict_patch.py for users who prefer bash.
# Forwards all args. Use --dry-run to validate without modifying source.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec python3 "$SCRIPT_DIR/apply_strict_patch.py" "$@"
