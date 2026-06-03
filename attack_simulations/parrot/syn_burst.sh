#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command hping3
print_config
echo "Scenario: TCP SYN burst"
echo "Expected features: many small TCP packets, SYN flags, high packet rate."

sudo hping3 -S -c "$COUNT" -i "u$INTERVAL_US" -p "$PORT" "$TARGET"
