#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command curl
print_config
echo "Scenario: HTTP request burst"
echo "Expected features: repeated TCP/HTTP flows, increased packets and byte rate."

url="http://$TARGET:$PORT$HTTP_PATH"
for i in $(seq 1 "$COUNT"); do
  curl -s --max-time 2 "$url" >/dev/null || true
done

echo "Completed $COUNT HTTP requests to $url"
