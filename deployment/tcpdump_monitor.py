import argparse
import json
import math
import re
import select
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from predict import load_metadata, predict_records


DEFAULT_FLOW_TIMEOUT = 3.0
DEFAULT_MIN_PACKETS = 5
DEFAULT_EVENTS_PATH = Path(__file__).resolve().parent / "events.jsonl"
PACKET_RE = re.compile(
    r"^(?P<ts>\d+(?:\.\d+)?)\s+IP\s+(?P<src>\S+)\s+>\s+(?P<dst>[^:]+):\s+(?P<body>.*)$"
)
LENGTH_RE = re.compile(r"\blength\s+(?P<length>\d+)\b")
FLAGS_RE = re.compile(r"Flags\s+\[(?P<flags>[^\]]+)\]")
WIN_RE = re.compile(r"\bwin\s+(?P<win>\d+)\b")


class FlowState:
    def __init__(self, first_seen, src, dst, sport, dport, proto):
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.src = src
        self.dst = dst
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.in_bytes = 0
        self.out_bytes = 0
        self.in_pkts = 0
        self.out_pkts = 0
        self.tcp_flags = 0
        self.client_tcp_flags = 0
        self.server_tcp_flags = 0
        self.tcp_win_max_in = 0
        self.tcp_win_max_out = 0
        self.packet_size_buckets = defaultdict(int)
        self.longest_pkt = 0
        self.shortest_pkt = None
        self.l7_proto = 0
        self.dns_query_id = 0
        self.dns_query_type = 0
        self.dns_ttl_answer = 0

    @property
    def total_packets(self):
        return self.in_pkts + self.out_pkts

    @property
    def total_bytes(self):
        return self.in_bytes + self.out_bytes


def split_host_port(endpoint):
    endpoint = endpoint.rstrip(":")
    parts = endpoint.split(".")
    if len(parts) >= 5 and all(part.isdigit() for part in parts[-5:]):
        return ".".join(parts[-5:-1]), int(parts[-1])
    return endpoint, 0


def infer_proto(body, sport, dport):
    if body.startswith("ICMP"):
        return 1
    if "Flags [" in body:
        return 6
    if "UDP" in body or sport == 53 or dport == 53:
        return 17
    return 0


def infer_l7(sport, dport):
    for port in (sport, dport):
        if port == 53:
            return 53
        if port in {80, 8080}:
            return 7
        if port == 443:
            return 91
        if port == 22:
            return 92
    return 0


def tcp_flag_value(flag_text):
    value = 0
    mapping = {
        "F": 1,
        "S": 2,
        "R": 4,
        "P": 8,
        ".": 16,
        "A": 16,
        "U": 32,
        "E": 64,
        "W": 128,
    }
    for char in flag_text:
        value |= mapping.get(char, 0)
    return value


def packet_bucket(size):
    if size <= 128:
        return "NUM_PKTS_UP_TO_128_BYTES"
    if size <= 256:
        return "NUM_PKTS_128_TO_256_BYTES"
    if size <= 512:
        return "NUM_PKTS_256_TO_512_BYTES"
    if size <= 1024:
        return "NUM_PKTS_512_TO_1024_BYTES"
    return "NUM_PKTS_1024_TO_1514_BYTES"


def parse_packet(line):
    match = PACKET_RE.match(line.strip())
    if not match:
        return None

    ts = float(match.group("ts"))
    src, sport = split_host_port(match.group("src"))
    dst, dport = split_host_port(match.group("dst"))
    body = match.group("body")
    proto = infer_proto(body, sport, dport)
    length_match = LENGTH_RE.search(body)
    payload_len = int(length_match.group("length")) if length_match else 0

    # tcpdump TCP "length" is payload length, so add a small header estimate.
    if proto == 6:
        size = max(payload_len + 40, 40)
    elif proto == 17:
        size = max(payload_len + 28, 28)
    elif proto == 1:
        size = max(payload_len + 20, 84)
    else:
        size = max(payload_len + 20, 60)

    flags_match = FLAGS_RE.search(body)
    flags = tcp_flag_value(flags_match.group("flags")) if flags_match else 0
    win_match = WIN_RE.search(body)
    win = int(win_match.group("win")) if win_match else 0

    return {
        "ts": ts,
        "src": src,
        "dst": dst,
        "sport": sport,
        "dport": dport,
        "proto": proto,
        "size": size,
        "flags": flags,
        "win": win,
        "l7_proto": infer_l7(sport, dport),
    }


def flow_key(packet, aggregate_hosts=False):
    if aggregate_hosts:
        forward = (packet["src"], packet["dst"], 0, packet["dport"], packet["proto"])
        reverse = (packet["dst"], packet["src"], 0, packet["sport"], packet["proto"])
        return min(forward, reverse)

    forward = (packet["src"], packet["dst"], packet["sport"], packet["dport"], packet["proto"])
    reverse = (packet["dst"], packet["src"], packet["dport"], packet["sport"], packet["proto"])
    return min(forward, reverse)


def update_flow(flow, packet):
    is_forward = packet["src"] == flow.src
    size = packet["size"]
    flow.last_seen = packet["ts"]

    if is_forward:
        flow.in_bytes += size
        flow.in_pkts += 1
        flow.client_tcp_flags |= packet["flags"]
        flow.tcp_win_max_in = max(flow.tcp_win_max_in, packet["win"])
    else:
        flow.out_bytes += size
        flow.out_pkts += 1
        flow.server_tcp_flags |= packet["flags"]
        flow.tcp_win_max_out = max(flow.tcp_win_max_out, packet["win"])

    flow.tcp_flags |= packet["flags"]
    flow.longest_pkt = max(flow.longest_pkt, size)
    flow.shortest_pkt = size if flow.shortest_pkt is None else min(flow.shortest_pkt, size)
    flow.packet_size_buckets[packet_bucket(size)] += 1
    flow.l7_proto = max(flow.l7_proto, packet["l7_proto"])


def flow_intensity(in_bytes, out_bytes, duration_ms):
    return math.log1p(in_bytes) + math.log1p(out_bytes) - math.log1p(duration_ms)


def flow_to_features(flow):
    duration_ms = max((flow.last_seen - flow.first_seen) * 1000.0, 1.0)
    duration_sec = duration_ms / 1000.0
    total_packets = flow.total_packets
    total_bytes = flow.total_bytes
    packets_safe = max(total_packets, 1)
    in_rate = flow.in_bytes / duration_sec
    out_rate = flow.out_bytes / duration_sec

    return {
        "PROTOCOL": flow.proto,
        "L7_PROTO": flow.l7_proto,
        "IN_BYTES": flow.in_bytes,
        "IN_PKTS": flow.in_pkts,
        "OUT_BYTES": flow.out_bytes,
        "OUT_PKTS": flow.out_pkts,
        "TCP_FLAGS": flow.tcp_flags,
        "CLIENT_TCP_FLAGS": flow.client_tcp_flags,
        "SERVER_TCP_FLAGS": flow.server_tcp_flags,
        "FLOW_DURATION_MILLISECONDS": duration_ms,
        "DURATION_IN": duration_ms if flow.in_pkts else 0,
        "DURATION_OUT": duration_ms if flow.out_pkts else 0,
        "MIN_TTL": 0,
        "MAX_TTL": 0,
        "LONGEST_FLOW_PKT": flow.longest_pkt,
        "SHORTEST_FLOW_PKT": flow.shortest_pkt or 0,
        "MIN_IP_PKT_LEN": flow.shortest_pkt or 0,
        "MAX_IP_PKT_LEN": flow.longest_pkt,
        "SRC_TO_DST_SECOND_BYTES": in_rate,
        "DST_TO_SRC_SECOND_BYTES": out_rate,
        "RETRANSMITTED_IN_BYTES": 0,
        "RETRANSMITTED_IN_PKTS": 0,
        "RETRANSMITTED_OUT_BYTES": 0,
        "RETRANSMITTED_OUT_PKTS": 0,
        "SRC_TO_DST_AVG_THROUGHPUT": in_rate * 8,
        "DST_TO_SRC_AVG_THROUGHPUT": out_rate * 8,
        "NUM_PKTS_UP_TO_128_BYTES": flow.packet_size_buckets["NUM_PKTS_UP_TO_128_BYTES"],
        "NUM_PKTS_128_TO_256_BYTES": flow.packet_size_buckets["NUM_PKTS_128_TO_256_BYTES"],
        "NUM_PKTS_256_TO_512_BYTES": flow.packet_size_buckets["NUM_PKTS_256_TO_512_BYTES"],
        "NUM_PKTS_512_TO_1024_BYTES": flow.packet_size_buckets["NUM_PKTS_512_TO_1024_BYTES"],
        "NUM_PKTS_1024_TO_1514_BYTES": flow.packet_size_buckets["NUM_PKTS_1024_TO_1514_BYTES"],
        "TCP_WIN_MAX_IN": flow.tcp_win_max_in,
        "TCP_WIN_MAX_OUT": flow.tcp_win_max_out,
        "ICMP_TYPE": 0,
        "ICMP_IPV4_TYPE": 0,
        "DNS_QUERY_ID": flow.dns_query_id,
        "DNS_QUERY_TYPE": flow.dns_query_type,
        "DNS_TTL_ANSWER": flow.dns_ttl_answer,
        "Total_Packets": total_packets,
        "Total_Bytes": total_bytes,
        "Packets_per_Second": total_packets / duration_sec,
        "Bytes_per_Packet": total_bytes / packets_safe,
        "Byte_Rate": total_bytes / duration_sec,
        "Flow_Intensity": flow_intensity(flow.in_bytes, flow.out_bytes, duration_ms),
    }


def append_event(payload, events_path):
    if not events_path:
        return
    path = Path(events_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def print_decision(flow, features, result, json_output=False, events_path=DEFAULT_EVENTS_PATH):
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "src": flow.src,
        "dst": flow.dst,
        "sport": flow.sport,
        "dport": flow.dport,
        "proto": flow.proto,
        "packets": features["Total_Packets"],
        "bytes": features["Total_Bytes"],
        "features": features,
        **result,
    }
    append_event(payload, events_path)

    if json_output:
        print(json.dumps(payload), flush=True)
        return

    label = "ANOMALY" if result["prediction"] == 1 else "BENIGN "
    print(
        f"[{payload['timestamp']}] [{label}] "
        f"{flow.src}:{flow.sport} -> {flow.dst}:{flow.dport} "
        f"proto={flow.proto} packets={payload['packets']} bytes={payload['bytes']} "
        f"score={result['score']:.6f} threshold={result['threshold']:.6f}",
        flush=True,
    )


def classify_flow(flow, json_output=False, events_path=DEFAULT_EVENTS_PATH):
    features = flow_to_features(flow)
    result = predict_records([features])[0]
    print_decision(flow, features, result, json_output, events_path)


def run_monitor(interface, flow_timeout, min_packets, bpf_filter, json_output, aggregate_hosts, events_path):
    metadata = load_metadata()
    command = ["tcpdump", "-tt", "-n", "-l", "-i", interface, bpf_filter]
    print(f"tcpdump monitor started on {interface}")
    print(f"Experiment: {metadata['experiment_id']} | Seed: {metadata['seed']}")
    print(f"Command: {' '.join(command)}")
    print(f"Flow timeout: {flow_timeout}s | Min packets: {min_packets}")
    print(f"Aggregate hosts: {aggregate_hosts}")
    print(f"Events path: {events_path}")
    print("Press Ctrl+C to stop.")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    flows = {}

    def flush_expired(now):
        for key, flow in list(flows.items()):
            if now - flow.last_seen < flow_timeout:
                continue
            flows.pop(key, None)
            if flow.total_packets >= min_packets:
                classify_flow(flow, json_output, events_path)

    try:
        while True:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                raise SystemExit(f"tcpdump stopped unexpectedly. {stderr.strip()}")

            readable, _, _ = select.select([process.stdout], [], [], 0.5)
            now = time.time()
            if readable:
                line = process.stdout.readline()
                packet = parse_packet(line)
                if packet:
                    packet["seen"] = time.time()
                    key = flow_key(packet, aggregate_hosts)
                    if key not in flows:
                        src, dst, sport, dport, proto = key
                        flows[key] = FlowState(packet["seen"], src, dst, sport, dport, proto)
                    update_flow(flows[key], packet)
                    now = packet["seen"]

            flush_expired(now)
    except KeyboardInterrupt:
        print("\nFlushing active flows...")
        for flow in list(flows.values()):
            if flow.total_packets >= min_packets:
                classify_flow(flow, json_output, events_path)
        print("tcpdump monitor stopped.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    parser = argparse.ArgumentParser(description="Read tcpdump output and classify real gateway traffic.")
    parser.add_argument("--interface", "-i", required=True, help="Gateway interface to monitor, for example eth1.")
    parser.add_argument("--flow-timeout", type=float, default=DEFAULT_FLOW_TIMEOUT)
    parser.add_argument("--min-packets", type=int, default=DEFAULT_MIN_PACKETS)
    parser.add_argument("--filter", default="ip", help="tcpdump BPF filter. Default: ip")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per classified flow.")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_PATH), help="JSONL event log path.")
    parser.add_argument(
        "--no-aggregate-hosts",
        action="store_true",
        help="Use full 5-tuple flows instead of grouping DDoS-like traffic by hosts and destination port.",
    )
    args = parser.parse_args()

    run_monitor(
        interface=args.interface,
        flow_timeout=args.flow_timeout,
        min_packets=args.min_packets,
        bpf_filter=args.filter,
        json_output=args.json,
        aggregate_hosts=not args.no_aggregate_hosts,
        events_path=args.events,
    )


if __name__ == "__main__":
    main()
