#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DNS_SERVER="${DNS_SERVER:-$TARGET}"
DOMAIN="${DOMAIN:-example.local}"
QUERIES="${QUERIES:-3}"
DELAY_SECONDS="${DELAY_SECONDS:-1}"

require_lab_target "$DNS_SERVER"
require_command dig
echo "DNS server: $DNS_SERVER"
echo "Domain: $DOMAIN"
echo "Queries: $QUERIES"
echo "Scenario: Benign DNS lookup"
echo "Expected features: very low UDP packet count, spaced queries, small byte volume."

for i in $(seq 1 "$QUERIES"); do
  dig @"$DNS_SERVER" "$DOMAIN" +time=1 +tries=1 >/dev/null || true
  sleep "$DELAY_SECONDS"
done

echo "Completed $QUERIES DNS lookup attempts."
