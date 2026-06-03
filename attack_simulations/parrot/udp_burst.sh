#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command hping3
print_config
echo "Scenario: UDP burst"
echo "Expected features: high packet rate, small stateless packets, elevated byte rate."

sudo hping3 --udp -c "$COUNT" -i "u$INTERVAL_US" -p "$PORT" --data 64 "$TARGET"
