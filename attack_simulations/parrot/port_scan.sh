#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PORT_RANGE="${PORT_RANGE:-1-1024}"
RATE="${RATE:-50}"

require_command nmap
print_config
echo "Port range: $PORT_RANGE"
echo "Scenario: TCP port scan"
echo "Expected features: many short TCP connection attempts across ports, mostly low byte flows."

nmap -sT -Pn --max-rate "$RATE" -p "$PORT_RANGE" "$TARGET"
