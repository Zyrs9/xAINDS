#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

REQUESTS="${REQUESTS:-5}"
DELAY_SECONDS="${DELAY_SECONDS:-1}"

require_command curl
print_config
echo "Requests: $REQUESTS"
echo "Delay seconds: $DELAY_SECONDS"
echo "Scenario: Benign HTTP browsing"
echo "Expected features: low request volume, spaced requests, moderate byte rate."

url="http://$TARGET:$PORT$HTTP_PATH"
for i in $(seq 1 "$REQUESTS"); do
  curl -s --max-time 3 "$url" >/dev/null || true
  sleep "$DELAY_SECONDS"
done

echo "Completed $REQUESTS spaced HTTP requests to $url"
