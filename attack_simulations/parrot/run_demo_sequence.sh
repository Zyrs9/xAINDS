#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

print_config
echo "Running bounded demo sequence against $TARGET"
echo

COUNT=150 INTERVAL_US=2000 "$SCRIPT_DIR/icmp_burst.sh"
sleep 3

COUNT=250 INTERVAL_US=1500 PORT="${PORT:-80}" "$SCRIPT_DIR/syn_burst.sh"
sleep 3

COUNT=80 PORT="${PORT:-80}" "$SCRIPT_DIR/http_burst.sh"
sleep 3

PORT_RANGE="${PORT_RANGE:-20-120}" RATE="${RATE:-25}" "$SCRIPT_DIR/port_scan.sh"

echo "Demo sequence finished."
