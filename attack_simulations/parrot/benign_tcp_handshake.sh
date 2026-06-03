#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CONNECTIONS="${CONNECTIONS:-5}"
DELAY_SECONDS="${DELAY_SECONDS:-1}"

require_command nc
print_config
echo "Connections: $CONNECTIONS"
echo "Delay seconds: $DELAY_SECONDS"
echo "Scenario: Benign low-rate TCP connection checks"
echo "Expected features: few TCP flows, low packet count, spaced attempts."

for i in $(seq 1 "$CONNECTIONS"); do
  nc -vz -w 2 "$TARGET" "$PORT" >/dev/null 2>&1 || true
  sleep "$DELAY_SECONDS"
done

echo "Completed $CONNECTIONS low-rate TCP checks."
