#!/bin/bash
# ==============================================================
# NIDS Benchmark — Traffic Replay Script
#
# Replays a PCAP file against both NVP and MVP systems using tcpreplay.
# This ensures both systems process IDENTICAL traffic for a fair comparison.
#
# Prerequisites:
#   sudo apt install tcpreplay
#
# Usage:
#   ./benchmark/traffic_replay.sh <pcap_file> <interface> [pps_rate]
#
# Example:
#   # Replay at 10,000 packets per second:
#   ./benchmark/traffic_replay.sh captures/test_traffic.pcap eth0 10000
#
#   # Replay at maximum speed:
#   ./benchmark/traffic_replay.sh captures/test_traffic.pcap eth0 0
# ==============================================================

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────
PCAP_FILE="${1:-}"
INTERFACE="${2:-eth0}"
PPS_RATE="${3:-10000}"   # Packets per second (0 = max speed)

if [ -z "$PCAP_FILE" ]; then
    echo ""
    echo "  NIDS Benchmark — Traffic Replay"
    echo "  ────────────────────────────────────────────"
    echo ""
    echo "  Usage: $0 <pcap_file> [interface] [pps_rate]"
    echo ""
    echo "  Arguments:"
    echo "    pcap_file    Path to the PCAP file to replay"
    echo "    interface    Network interface (default: eth0)"
    echo "    pps_rate     Packets/sec rate (default: 10000, 0=max)"
    echo ""
    echo "  Example:"
    echo "    $0 captures/test.pcap eth0 10000"
    echo ""
    exit 1
fi

if [ ! -f "$PCAP_FILE" ]; then
    echo "[✘] PCAP file not found: $PCAP_FILE"
    exit 1
fi

# ── Check Dependencies ───────────────────────────────────────
if ! command -v tcpreplay &> /dev/null; then
    echo "[✘] tcpreplay not installed. Run: sudo apt install tcpreplay"
    exit 1
fi

# ── PCAP Info ─────────────────────────────────────────────────
echo ""
echo "  ══════════════════════════════════════════════════"
echo "  NIDS BENCHMARK — Traffic Replay"
echo "  ══════════════════════════════════════════════════"
echo "  PCAP:        $PCAP_FILE"
echo "  Interface:   $INTERFACE"
echo "  PPS Rate:    ${PPS_RATE} pps"
echo ""

# Show PCAP stats
if command -v capinfos &> /dev/null; then
    echo "  ── PCAP Statistics ──"
    capinfos -c -s -u "$PCAP_FILE" 2>/dev/null | tail -5 | sed 's/^/  /'
    echo ""
fi

echo "  Starting replay in 3 seconds..."
sleep 3

# ── Replay ────────────────────────────────────────────────────
echo ""
echo "  [*] Replaying traffic..."
echo ""

if [ "$PPS_RATE" -eq 0 ] 2>/dev/null; then
    # Maximum speed replay
    sudo tcpreplay \
        --intf1="$INTERFACE" \
        --topspeed \
        --stats=5 \
        "$PCAP_FILE"
else
    # Rate-limited replay
    sudo tcpreplay \
        --intf1="$INTERFACE" \
        --pps="$PPS_RATE" \
        --stats=5 \
        "$PCAP_FILE"
fi

echo ""
echo "  [✔] Replay complete."
echo ""
