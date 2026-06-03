#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command ping
print_config
echo "Scenario: Benign connectivity check"
echo "Expected features: low packet count, regular ICMP timing, low byte rate."

ping -c "${COUNT:-5}" -i "${INTERVAL_SECONDS:-1}" "$TARGET"
