import argparse
import json
import time
from collections import defaultdict

from predict import load_metadata, predict_records

try:
    from scapy.all import DNS, IP, TCP, UDP, sniff
except ImportError as exc:
    raise SystemExit("scapy is required. Install it with: pip install -r requirements.txt") from exc


DEFAULT_FLOW_TIMEOUT = 10.0
DEFAULT_MIN_PACKETS = 5


class FlowState:
    def __init__(self, first_packet_time, src, dst, sport, dport, proto):
        self.first_seen = first_packet_time
        self.last_seen = first_packet_time
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
        self.min_ttl = None
        self.max_ttl = None
        self.longest_pkt = 0
        self.shortest_pkt = None
        self.min_ip_pkt_len = None
        self.max_ip_pkt_len = 0
        self.dns_query_id = 0
        self.dns_query_type = 0
        self.dns_ttl_answer = 0
        self.l7_proto = 0

    @property
    def total_packets(self):
        return self.in_pkts + self.out_pkts

    @property
    def total_bytes(self):
        return self.in_bytes + self.out_bytes


def packet_proto(packet):
    if TCP in packet:
        return 6
    if UDP in packet:
        return 17
    return int(packet[IP].proto)


def packet_ports(packet):
    if TCP in packet:
        return int(packet[TCP].sport), int(packet[TCP].dport)
    if UDP in packet:
        return int(packet[UDP].sport), int(packet[UDP].dport)
    return 0, 0


def flow_key(src, dst, sport, dport, proto):
    forward = (src, dst, sport, dport, proto)
    reverse = (dst, src, dport, sport, proto)
    return min(forward, reverse)


def l7_proto(packet):
    if DNS in packet:
        return 53
    if TCP in packet:
        dport = int(packet[TCP].dport)
        sport = int(packet[TCP].sport)
        for port in (dport, sport):
            if port in {80, 8080}:
                return 7
            if port == 443:
                return 91
            if port == 22:
                return 92
    return 0


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


def update_flow(flow, packet, now):
    src = packet[IP].src
    size = int(len(packet))
    ip_len = int(getattr(packet[IP], "len", size) or size)
    ttl = int(getattr(packet[IP], "ttl", 0) or 0)
    is_forward = src == flow.src

    flow.last_seen = now
    if is_forward:
        flow.in_bytes += size
        flow.in_pkts += 1
    else:
        flow.out_bytes += size
        flow.out_pkts += 1

    flow.min_ttl = ttl if flow.min_ttl is None else min(flow.min_ttl, ttl)
    flow.max_ttl = max(flow.max_ttl or ttl, ttl)
    flow.longest_pkt = max(flow.longest_pkt, size)
    flow.shortest_pkt = size if flow.shortest_pkt is None else min(flow.shortest_pkt, size)
    flow.min_ip_pkt_len = ip_len if flow.min_ip_pkt_len is None else min(flow.min_ip_pkt_len, ip_len)
    flow.max_ip_pkt_len = max(flow.max_ip_pkt_len, ip_len)
    flow.packet_size_buckets[packet_bucket(size)] += 1
    flow.l7_proto = max(flow.l7_proto, l7_proto(packet))

    if TCP in packet:
        flags = int(packet[TCP].flags)
        win = int(packet[TCP].window)
        flow.tcp_flags |= flags
        if is_forward:
            flow.client_tcp_flags |= flags
            flow.tcp_win_max_in = max(flow.tcp_win_max_in, win)
        else:
            flow.server_tcp_flags |= flags
            flow.tcp_win_max_out = max(flow.tcp_win_max_out, win)

    if DNS in packet:
        dns = packet[DNS]
        flow.dns_query_id = int(getattr(dns, "id", 0) or 0)
        if getattr(dns, "qd", None):
            flow.dns_query_type = int(getattr(dns.qd, "qtype", 0) or 0)
        if getattr(dns, "an", None):
            flow.dns_ttl_answer = int(getattr(dns.an, "ttl", 0) or 0)


def flow_to_features(flow):
    duration_ms = max((flow.last_seen - flow.first_seen) * 1000.0, 1.0)
    duration_sec = duration_ms / 1000.0
    total_packets = flow.total_packets
    total_bytes = flow.total_bytes
    packets_safe = max(total_packets, 1)
    in_rate = flow.in_bytes / duration_sec
    out_rate = flow.out_bytes / duration_sec

    record = {
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
        "MIN_TTL": flow.min_ttl or 0,
        "MAX_TTL": flow.max_ttl or 0,
        "LONGEST_FLOW_PKT": flow.longest_pkt,
        "SHORTEST_FLOW_PKT": flow.shortest_pkt or 0,
        "MIN_IP_PKT_LEN": flow.min_ip_pkt_len or 0,
        "MAX_IP_PKT_LEN": flow.max_ip_pkt_len,
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
    return record


def flow_intensity(in_bytes, out_bytes, duration_ms):
    import math

    return math.log1p(in_bytes) + math.log1p(out_bytes) - math.log1p(duration_ms)


def print_decision(flow, features, result, json_output=False):
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "src": flow.src,
        "dst": flow.dst,
        "sport": flow.sport,
        "dport": flow.dport,
        "proto": flow.proto,
        "packets": features["Total_Packets"],
        "bytes": features["Total_Bytes"],
        **result,
    }
    if json_output:
        print(json.dumps(payload), flush=True)
        return

    prefix = "ANOMALY" if result["prediction"] == 1 else "BENIGN "
    print(
        f"[{payload['timestamp']}] [{prefix}] "
        f"{flow.src}:{flow.sport} -> {flow.dst}:{flow.dport} "
        f"proto={flow.proto} packets={payload['packets']} bytes={payload['bytes']} "
        f"score={result['score']:.6f} threshold={result['threshold']:.6f}",
        flush=True,
    )


def run_monitor(interface, flow_timeout, min_packets, bpf_filter, json_output):
    metadata = load_metadata()
    print(f"Live monitor started on {interface}")
    print(f"Experiment: {metadata['experiment_id']} | Seed: {metadata['seed']}")
    print(f"Flow timeout: {flow_timeout}s | Min packets: {min_packets}")
    print("Press Ctrl+C to stop.")

    flows = {}

    def flush_expired(now):
        expired = [
            key for key, flow in flows.items()
            if now - flow.last_seen >= flow_timeout and flow.total_packets >= min_packets
        ]
        for key in expired:
            flow = flows.pop(key)
            features = flow_to_features(flow)
            result = predict_records([features])[0]
            print_decision(flow, features, result, json_output)

        stale = [
            key for key, flow in flows.items()
            if now - flow.last_seen >= flow_timeout and flow.total_packets < min_packets
        ]
        for key in stale:
            flows.pop(key, None)

    def handle_packet(packet):
        if IP not in packet:
            return

        now = time.time()
        src = packet[IP].src
        dst = packet[IP].dst
        proto = packet_proto(packet)
        sport, dport = packet_ports(packet)
        key = flow_key(src, dst, sport, dport, proto)

        if key not in flows:
            f_src, f_dst, f_sport, f_dport, f_proto = key
            flows[key] = FlowState(now, f_src, f_dst, f_sport, f_dport, f_proto)

        flow = flows[key]
        update_flow(flow, packet, now)
        flush_expired(now)

    try:
        sniff(iface=interface, filter=bpf_filter, prn=handle_packet, store=False)
    except PermissionError as exc:
        raise SystemExit("Permission denied. Run with sudo.") from exc
    except KeyboardInterrupt:
        now = time.time() + flow_timeout
        for key in list(flows.keys()):
            flow = flows[key]
            if flow.total_packets >= min_packets:
                features = flow_to_features(flow)
                result = predict_records([features])[0]
                print_decision(flow, features, result, json_output)
        print("Live monitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Capture live traffic and classify flows with the NIDS model.")
    parser.add_argument("--interface", "-i", required=True, help="Network interface to monitor, for example eth1.")
    parser.add_argument("--flow-timeout", type=float, default=DEFAULT_FLOW_TIMEOUT)
    parser.add_argument("--min-packets", type=int, default=DEFAULT_MIN_PACKETS)
    parser.add_argument("--filter", default="ip", help="BPF capture filter. Default: ip")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per classified flow.")
    args = parser.parse_args()

    run_monitor(
        interface=args.interface,
        flow_timeout=args.flow_timeout,
        min_packets=args.min_packets,
        bpf_filter=args.filter,
        json_output=args.json,
    )


if __name__ == "__main__":
    main()
