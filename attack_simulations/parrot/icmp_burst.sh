#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command hping3
print_config
echo "Scenario: ICMP burst"
echo "Expected features: repeated ICMP packets, high packets per second, low payload diversity."

sudo hping3 --icmp -c "$COUNT" -i "u$INTERVAL_US" "$TARGET"
