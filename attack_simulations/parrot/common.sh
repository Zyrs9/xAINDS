#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-10.10.20.10}"
PORT="${PORT:-80}"
COUNT="${COUNT:-500}"
INTERVAL_US="${INTERVAL_US:-1000}"
HTTP_PATH="${HTTP_PATH:-/}"

require_lab_target() {
  local target="$1"
  if [[ ! "$target" =~ ^10\.10\.20\.[0-9]{1,3}$ ]]; then
    echo "Refusing target outside victim lab network: $target" >&2
    echo "Use a Debian victim IP such as 10.10.20.10 or 10.10.20.11." >&2
    exit 2
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 3
  fi
}

print_config() {
  echo "Target: $TARGET"
  echo "Port: $PORT"
  echo "Count: $COUNT"
}

require_lab_target "$TARGET"
