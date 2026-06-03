#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

print_config
echo "Running benign demo sequence against $TARGET"
echo

COUNT=5 INTERVAL_SECONDS=1 "$SCRIPT_DIR/benign_ping_check.sh"
sleep 2

REQUESTS=5 DELAY_SECONDS=1 PORT="${PORT:-8080}" "$SCRIPT_DIR/benign_http_browse.sh"
sleep 2

CONNECTIONS=5 DELAY_SECONDS=1 PORT="${PORT:-80}" "$SCRIPT_DIR/benign_tcp_handshake.sh"

echo "Benign demo sequence finished."
