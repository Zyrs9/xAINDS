"""
eBPF User-Space Listener — BCC Python Bridge

This daemon attaches the eBPF/XDP program to the network interface,
polls the perf event ring buffer for completed flow records, and
converts C struct data into numpy arrays matching the 10-feature
UNSW-NB15 format for the ML engine.

Architecture:
    ┌──────────────┐     perf event     ┌───────────────┐     numpy      ┌──────────┐
    │  XDP Kernel  │ ──────────────────→ │  user_space   │ ────────────→ │ ML Engine│
    │  (C/eBPF)    │    flow_export      │  (Python/BCC) │   10-feature  │          │
    └──────────────┘                     └───────────────┘    array       └──────────┘

Requirements:
    - Linux kernel ≥ 4.15 (XDP support)
    - BCC (apt install bpfcc-tools python3-bpfcc)
    - Root/CAP_NET_ADMIN privileges

Usage:
    sudo python -m src.ebpf.user_space --interface eth0
    sudo python -m src.ebpf.user_space --interface eth0 --api http://localhost:8000

Author: NIDS Research Team
"""

import sys
import os
import time
import json
import struct
import signal
import logging
import argparse
import ctypes as ct
from pathlib import Path
from typing import Optional, Callable
import threading
import queue

# Fix Windows terminal encoding
if sys.platform == "win32":
    os.system("")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# BCC import — only available on Linux with BCC installed
try:
    from bcc import BPF
    BCC_AVAILABLE = True
except ImportError:
    BCC_AVAILABLE = False

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ml.engine import NIDSEngine, CANONICAL_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("nids.ebpf")


# ---------------------------------------------------------------------------
# C Struct Definitions (must mirror flow_tracker.c exactly)
# ---------------------------------------------------------------------------

class FlowKey(ct.Structure):
    """Mirrors `struct flow_key` from flow_tracker.c"""
    _fields_ = [
        ("ip_lo",    ct.c_uint32),
        ("ip_hi",    ct.c_uint32),
        ("port_lo",  ct.c_uint16),
        ("port_hi",  ct.c_uint16),
        ("protocol", ct.c_uint8),
        ("pad",      ct.c_uint8 * 3),
    ]


class FlowExport(ct.Structure):
    """Mirrors `struct flow_export` from flow_tracker.c"""
    _fields_ = [
        # Flow key
        ("key", FlowKey),

        # 10 ML features
        ("min_ttl",                  ct.c_uint32),
        ("max_ttl",                  ct.c_uint32),
        ("shortest_flow_pkt",        ct.c_uint32),
        ("longest_flow_pkt",         ct.c_uint32),
        ("min_ip_pkt_len",           ct.c_uint32),
        ("max_ip_pkt_len",           ct.c_uint32),
        ("out_bytes",                ct.c_uint64),
        ("out_pkts",                 ct.c_uint32),
        ("dst_to_src_second_bytes",  ct.c_uint64),
        ("num_pkts_up_to_128",       ct.c_uint32),

        # Metadata
        ("total_pkts",               ct.c_uint32),
        ("flow_duration_ns",         ct.c_uint64),
    ]


def ip_to_str(ip_int: int) -> str:
    """Convert a network-byte-order u32 IP to dotted-quad string."""
    return f"{ip_int & 0xFF}.{(ip_int >> 8) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 24) & 0xFF}"


def proto_to_str(proto: int) -> str:
    """Convert IP protocol number to name."""
    return {6: "TCP", 17: "UDP"}.get(proto, f"PROTO_{proto}")


# ---------------------------------------------------------------------------
# Flow Event Handler
# ---------------------------------------------------------------------------

class FlowProcessor:
    """
    Receives flow export events from eBPF and processes them through the
    ML engine for real-time anomaly detection.
    """

    def __init__(self, engine: NIDSEngine, api_url: Optional[str] = None):
        self.engine = engine
        self.api_url = api_url
        self.flows_processed = 0
        self.anomalies_detected = 0

        # Queued background worker for Fast API forwarding
        if api_url:
            self._queue = queue.Queue(maxsize=100000)
            self._api_worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._api_worker.start()
        else:
            self._queue = None

    def _worker_loop(self):
        try:
            import requests
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
        except ImportError:
            logger.warning("requests library not available. API forwarding disabled.")
            return

        while True:
            try:
                payload = self._queue.get()
                session.post(
                    f"{self.api_url}/analyze",
                    json=payload,
                    timeout=2,
                )
                self._queue.task_done()
            except Exception:
                # Silently ignore network errors to keep loop alive
                pass

    def flow_to_numpy(self, event: FlowExport) -> np.ndarray:
        """
        Convert a C struct FlowExport to a numpy array in CANONICAL_FEATURES order.

        The order MUST match the training pipeline's feature column order.
        """
        return np.array([
            float(event.min_ttl),                  # MIN_TTL
            float(event.max_ttl),                  # MAX_TTL
            float(event.shortest_flow_pkt),        # SHORTEST_FLOW_PKT
            float(event.longest_flow_pkt),         # LONGEST_FLOW_PKT
            float(event.min_ip_pkt_len),           # MIN_IP_PKT_LEN
            float(event.max_ip_pkt_len),           # MAX_IP_PKT_LEN
            float(event.out_bytes),                # OUT_BYTES
            float(event.out_pkts),                 # OUT_PKTS
            float(event.dst_to_src_second_bytes),  # DST_TO_SRC_SECOND_BYTES
            float(event.num_pkts_up_to_128),       # NUM_PKTS_UP_TO_128_BYTES
        ], dtype=np.float64)

    def flow_to_dict(self, event: FlowExport) -> dict:
        """Convert flow export to a feature dictionary."""
        return {
            "MIN_TTL": float(event.min_ttl),
            "MAX_TTL": float(event.max_ttl),
            "SHORTEST_FLOW_PKT": float(event.shortest_flow_pkt),
            "LONGEST_FLOW_PKT": float(event.longest_flow_pkt),
            "MIN_IP_PKT_LEN": float(event.min_ip_pkt_len),
            "MAX_IP_PKT_LEN": float(event.max_ip_pkt_len),
            "OUT_BYTES": float(event.out_bytes),
            "OUT_PKTS": float(event.out_pkts),
            "DST_TO_SRC_SECOND_BYTES": float(event.dst_to_src_second_bytes),
            "NUM_PKTS_UP_TO_128_BYTES": float(event.num_pkts_up_to_128),
        }

    def handle_event(self, cpu: int, data: ct.c_void_p, size: int):
        """
        BCC perf event callback. Called for each completed flow.

        Args:
            cpu: CPU core that generated the event.
            data: Pointer to raw event data.
            size: Size of the event data.
        """
        event = ct.cast(data, ct.POINTER(FlowExport)).contents

        # Convert to numpy and feature dict
        feature_vector = self.flow_to_numpy(event)
        feature_dict = self.flow_to_dict(event)

        # Run ML inference
        result = self.engine.analyze(feature_vector, feature_dict=feature_dict)
        self.flows_processed += 1

        # Format flow identification
        flow_id = (
            f"{ip_to_str(event.key.ip_lo)}:{event.key.port_lo} ↔ "
            f"{ip_to_str(event.key.ip_hi)}:{event.key.port_hi} "
            f"({proto_to_str(event.key.protocol)})"
        )
        duration_ms = event.flow_duration_ns / 1_000_000

        # Color-coded terminal output
        if result.alert_level == "CRITICAL":
            color = "\033[91m"  # Red
            self.anomalies_detected += 1
        elif result.alert_level == "WARNING":
            color = "\033[93m"  # Yellow
            self.anomalies_detected += 1
        else:
            color = "\033[92m"  # Green
        reset = "\033[0m"

        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[{timestamp}] {color}{result.alert_level:8s}{reset} │ "
            f"Score: {result.anomaly_score:+.4f} │ "
            f"Pkts: {event.total_pkts:4d} │ "
            f"Dur: {duration_ms:7.1f}ms │ "
            f"{flow_id}"
        )

        # Forward to API via background thread
        if self._queue is not None:
            try:
                self._queue.put_nowait(result.to_dict())
            except queue.Full:
                pass  # Drop payload if API is too slow, protecting eBPF memory

    def print_stats(self):
        """Print processing statistics."""
        print(f"\n--- Session Statistics ---")
        print(f"  Flows processed:   {self.flows_processed}")
        print(f"  Anomalies found:   {self.anomalies_detected}")
        if self.flows_processed > 0:
            rate = self.anomalies_detected / self.flows_processed * 100
            print(f"  Anomaly rate:      {rate:.1f}%")


# ---------------------------------------------------------------------------
# eBPF Loader & Main Loop
# ---------------------------------------------------------------------------

# Inline BCC version of the eBPF C program
# BCC compiles C at runtime — we read the .c file directly
BPF_C_SOURCE_PATH = Path(__file__).parent / "flow_tracker.c"


def load_bpf_program(interface: str) -> BPF:
    """
    Load and attach the eBPF/XDP program to the specified interface.

    BCC compiles the C source at runtime and injects it into the kernel.
    """
    if not BCC_AVAILABLE:
        raise RuntimeError(
            "BCC (BPF Compiler Collection) is not installed. "
            "Install with: sudo apt install bpfcc-tools python3-bpfcc"
        )

    # For BCC, we need to read the C source and adapt includes
    c_source = BPF_C_SOURCE_PATH.read_text()

    # BCC uses its own headers — replace standard libbpf includes
    bcc_source = c_source.replace('#include <linux/bpf.h>', '')
    bcc_source = bcc_source.replace('#include <bpf/bpf_helpers.h>', '')
    bcc_source = bcc_source.replace('#include <bpf/bpf_endian.h>', '')
    # BCC provides bpf_htons, bpf_ntohs as builtins

    logger.info(f"Compiling eBPF program ({len(c_source)} bytes)...")
    b = BPF(text=bcc_source)

    # Attach XDP program to the interface
    fn = b.load_func("xdp_flow_tracker", BPF.XDP)
    b.attach_xdp(interface, fn, 0)
    logger.info(f"XDP program attached to interface: {interface}")

    return b


def run_listener(interface: str, artifacts_dir: str = ".",
                 api_url: Optional[str] = None):
    """
    Main event loop: attach eBPF, process flow events, run ML inference.

    Args:
        interface: Network interface to attach XDP program to (e.g., "eth0").
        artifacts_dir: Path to model .pkl artifacts.
        api_url: Optional FastAPI URL for forwarding results.
    """
    print("=" * 70)
    print("  eBPF/XDP FLOW TRACKER — Real-Time Network Intrusion Detection")
    print("=" * 70)
    print(f"  Interface:     {interface}")
    print(f"  Artifacts:     {artifacts_dir}")
    print(f"  API Forward:   {api_url or 'Disabled'}")
    print(f"  Press Ctrl+C to stop.")
    print("=" * 70)
    print()

    # Initialize ML engine
    engine = NIDSEngine(artifacts_dir=artifacts_dir)
    engine.load_artifacts()

    # Initialize flow processor
    processor = FlowProcessor(engine=engine, api_url=api_url)

    # Load and attach eBPF program
    bpf = load_bpf_program(interface)

    # Register the perf event callback
    bpf["flow_events"].open_perf_buffer(processor.handle_event, page_cnt=64)

    # Print table header
    print(f"{'Time':^10s} │ {'Level':^8s} │ {'Score':^12s} │ "
          f"{'Pkts':^6s} │ {'Duration':^10s} │ Flow")
    print("─" * 90)

    # Graceful shutdown handler
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main polling loop
    try:
        while running:
            bpf.perf_buffer_poll(timeout=100)
    except KeyboardInterrupt:
        pass
    finally:
        # Detach XDP program
        logger.info("Detaching XDP program...")
        bpf.remove_xdp(interface, 0)
        processor.print_stats()
        print("\neBPF listener stopped cleanly.")


# ---------------------------------------------------------------------------
# Offline Mode (for testing on non-Linux / without eBPF)
# ---------------------------------------------------------------------------

def run_offline_test(artifacts_dir: str = "."):
    """
    Test the flow processing pipeline without eBPF.
    Uses synthetic flow data for validation on Windows/macOS.
    """
    print("=" * 70)
    print("  OFFLINE TEST MODE (No eBPF — synthetic flow data)")
    print("=" * 70)
    print()

    engine = NIDSEngine(artifacts_dir=artifacts_dir)
    engine.load_artifacts()

    processor = FlowProcessor(engine=engine)

    # Simulate some flows
    test_flows = [
        # Normal web browsing
        {"MIN_TTL": 64, "MAX_TTL": 64, "SHORTEST_FLOW_PKT": 54,
         "LONGEST_FLOW_PKT": 1500, "MIN_IP_PKT_LEN": 40,
         "MAX_IP_PKT_LEN": 1480, "OUT_BYTES": 2500, "OUT_PKTS": 15,
         "DST_TO_SRC_SECOND_BYTES": 45000, "NUM_PKTS_UP_TO_128_BYTES": 8},
        # Suspicious: high outbound
        {"MIN_TTL": 12, "MAX_TTL": 245, "SHORTEST_FLOW_PKT": 40,
         "LONGEST_FLOW_PKT": 1500, "MIN_IP_PKT_LEN": 20,
         "MAX_IP_PKT_LEN": 1480, "OUT_BYTES": 985000, "OUT_PKTS": 1200,
         "DST_TO_SRC_SECOND_BYTES": 4500, "NUM_PKTS_UP_TO_128_BYTES": 950},
        # Suspicious: small packet flood
        {"MIN_TTL": 128, "MAX_TTL": 128, "SHORTEST_FLOW_PKT": 60,
         "LONGEST_FLOW_PKT": 60, "MIN_IP_PKT_LEN": 40,
         "MAX_IP_PKT_LEN": 40, "OUT_BYTES": 60000, "OUT_PKTS": 1000,
         "DST_TO_SRC_SECOND_BYTES": 0, "NUM_PKTS_UP_TO_128_BYTES": 1000},
    ]

    print(f"{'Time':^10s} │ {'Level':^8s} │ {'Score':^12s} │ Flow Description")
    print("─" * 70)

    labels = ["Normal browsing", "Data exfiltration pattern", "UDP flood pattern"]

    for flow_dict, label in zip(test_flows, labels):
        feature_vector = np.array([flow_dict[f] for f in CANONICAL_FEATURES],
                                  dtype=np.float64)
        result = engine.analyze(feature_vector, feature_dict=flow_dict)

        if result.alert_level == "CRITICAL":
            color = "\033[91m"
        elif result.alert_level == "WARNING":
            color = "\033[93m"
        else:
            color = "\033[92m"
        reset = "\033[0m"

        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[{timestamp}] {color}{result.alert_level:8s}{reset} │ "
            f"Score: {result.anomaly_score:+.4f} │ "
            f"{label}"
        )

    print("\nOffline test complete.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="eBPF/XDP Flow Tracker — NIDS Data Plane Listener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode (Linux with BCC, requires root):
  sudo python -m src.ebpf.user_space --interface eth0

  # With API forwarding:
  sudo python -m src.ebpf.user_space --interface eth0 --api http://localhost:8000

  # Offline test mode (any OS):
  python -m src.ebpf.user_space --offline
        """
    )
    parser.add_argument(
        "--interface", "-i", type=str, default="eth0",
        help="Network interface to attach XDP program to (default: eth0)"
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default=".",
        help="Directory containing .pkl model artifacts"
    )
    parser.add_argument(
        "--api", type=str, default=None,
        help="FastAPI URL for forwarding results (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run in offline test mode (no eBPF, synthetic flows)"
    )

    args = parser.parse_args()

    if args.offline:
        run_offline_test(artifacts_dir=args.artifacts_dir)
    else:
        run_listener(
            interface=args.interface,
            artifacts_dir=args.artifacts_dir,
            api_url=args.api,
        )


if __name__ == "__main__":
    main()
