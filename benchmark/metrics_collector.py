"""
NIDS Benchmark — CPU/RAM Metrics Collector

Collects system resource metrics at sub-second intervals for side-by-side
comparison between the NVP (Scapy) and MVP (eBPF) architectures.

Output:
    CSV file with columns: timestamp, cpu_percent, ram_mb, ram_percent
    Suitable for direct import into matplotlib/LaTeX pgfplots for the
    IEEE paper's performance comparison figures (Section V).

Usage:
    # Monitor current system for 60 seconds:
    python benchmark/metrics_collector.py --duration 60 --output nvp_metrics.csv

    # Monitor a specific process:
    python benchmark/metrics_collector.py --pid 12345 --duration 60

    # Side-by-side benchmark (run on each system separately):
    # Terminal 1 (NVP VM):
    python benchmark/metrics_collector.py --duration 120 --output results/nvp_cpu.csv
    # Terminal 2 (MVP VM):
    python benchmark/metrics_collector.py --duration 120 --output results/mvp_cpu.csv

Author: NIDS Research Team
"""

import os
import sys
import csv
import time
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    print("[✘] psutil is required: pip install psutil")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("nids.benchmark")


# ---------------------------------------------------------------------------
# Colors for terminal output
# ---------------------------------------------------------------------------

class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def cpu_bar(pct: float, width: int = 30) -> str:
    """Render a colored CPU usage bar."""
    filled = int(pct / 100 * width)
    empty = width - filled

    if pct >= 80:
        color = C.RED
    elif pct >= 50:
        color = C.YELLOW
    else:
        color = C.GREEN

    return f"{color}{'█' * filled}{'░' * empty}{C.RESET} {pct:5.1f}%"


def ram_bar(pct: float, width: int = 15) -> str:
    """Render a colored RAM usage bar."""
    filled = int(pct / 100 * width)
    empty = width - filled

    if pct >= 80:
        color = C.RED
    elif pct >= 50:
        color = C.YELLOW
    else:
        color = C.GREEN

    return f"{color}{'█' * filled}{'░' * empty}{C.RESET} {pct:5.1f}%"


# ---------------------------------------------------------------------------
# System-wide Collector
# ---------------------------------------------------------------------------

def collect_system_metrics(duration: float, interval: float,
                           output_path: str, label: str = "system"):
    """
    Collect system-wide CPU and RAM metrics.

    Args:
        duration: Total collection time in seconds (0 = infinite).
        interval: Sampling interval in seconds.
        output_path: CSV output file path.
        label: Label for the metrics run (e.g., "NVP" or "MVP").
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  NIDS BENCHMARK — System Metrics Collector{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
    print(f"  Label:       {label}")
    print(f"  Duration:    {duration}s" if duration > 0 else "  Duration:    Infinite (Ctrl+C to stop)")
    print(f"  Interval:    {interval}s")
    print(f"  Output:      {output_file}")
    print(f"  CPU Cores:   {psutil.cpu_count(logical=True)}")
    print(f"  Total RAM:   {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"{C.BOLD}{C.CYAN}{'=' * 70}{C.RESET}")
    print()

    # CSV setup
    fieldnames = [
        "timestamp", "elapsed_seconds", "label",
        "cpu_percent", "cpu_per_core",
        "ram_used_mb", "ram_percent", "ram_available_mb"
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Header for live display
        print(f"  {'Time':^10s} │ {'Elapsed':^8s} │ {'CPU':^38s} │ {'RAM':^24s} │ {'RAM MB':>8s}")
        print(f"  {'─' * 10} │ {'─' * 8} │ {'─' * 38} │ {'─' * 24} │ {'─' * 8}")

        running = True
        start_time = time.time()
        samples = 0

        def stop_handler(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, stop_handler)

        # Prime the CPU counter (first call always returns 0)
        psutil.cpu_percent(interval=None)

        while running:
            now = time.time()
            elapsed = now - start_time

            if duration > 0 and elapsed >= duration:
                break

            # Collect metrics
            cpu_pct = psutil.cpu_percent(interval=None)
            per_core = psutil.cpu_percent(percpu=True)
            mem = psutil.virtual_memory()
            ram_used_mb = mem.used / (1024 ** 2)
            ram_available_mb = mem.available / (1024 ** 2)

            # Write CSV row
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "label": label,
                "cpu_percent": round(cpu_pct, 1),
                "cpu_per_core": str([round(c, 1) for c in per_core]),
                "ram_used_mb": round(ram_used_mb, 1),
                "ram_percent": round(mem.percent, 1),
                "ram_available_mb": round(ram_available_mb, 1),
            })
            csvfile.flush()
            samples += 1

            # Live terminal display
            timestamp = time.strftime("%H:%M:%S")
            print(
                f"  {timestamp:^10s} │ {elapsed:7.1f}s │ "
                f"{cpu_bar(cpu_pct)} │ "
                f"{ram_bar(mem.percent)} │ "
                f"{ram_used_mb:7.0f}M",
                end="\r" if sys.stdout.isatty() else "\n"
            )

            # Sleep until next sample
            sleep_time = max(0, interval - (time.time() - now))
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Summary
    print()
    print()
    print(f"{C.BOLD}  ── Collection Complete ──{C.RESET}")
    print(f"  Samples:   {samples}")
    print(f"  Duration:  {time.time() - start_time:.1f}s")
    print(f"  Output:    {output_file.resolve()}")
    print()


# ---------------------------------------------------------------------------
# Process-specific Collector
# ---------------------------------------------------------------------------

def collect_process_metrics(pid: int, duration: float, interval: float,
                            output_path: str, label: str = "process"):
    """
    Collect metrics for a specific process (by PID).
    Useful for isolating NIDS resource consumption.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.error(f"Process {pid} not found.")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Monitoring PID {pid}: {proc.name()}")
    print(f"  Output: {output_file}\n")

    fieldnames = [
        "timestamp", "elapsed_seconds", "label", "pid",
        "cpu_percent", "ram_rss_mb", "ram_vms_mb",
        "threads", "status"
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        running = True
        start_time = time.time()

        def stop_handler(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, stop_handler)

        while running:
            now = time.time()
            elapsed = now - start_time

            if duration > 0 and elapsed >= duration:
                break

            try:
                cpu_pct = proc.cpu_percent(interval=None)
                mem_info = proc.memory_info()

                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "label": label,
                    "pid": pid,
                    "cpu_percent": round(cpu_pct, 1),
                    "ram_rss_mb": round(mem_info.rss / (1024 ** 2), 1),
                    "ram_vms_mb": round(mem_info.vms / (1024 ** 2), 1),
                    "threads": proc.num_threads(),
                    "status": proc.status(),
                })
                csvfile.flush()

                timestamp = time.strftime("%H:%M:%S")
                rss = mem_info.rss / (1024 ** 2)
                print(
                    f"  [{timestamp}] CPU: {cpu_pct:5.1f}% │ "
                    f"RSS: {rss:7.1f} MB │ "
                    f"Threads: {proc.num_threads()}",
                    end="\r"
                )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning(f"Process {pid} terminated or access denied.")
                break

            time.sleep(interval)

    print(f"\n\n  Collection complete: {output_file.resolve()}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NIDS Benchmark — System Resource Metrics Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # System-wide, 2 minutes, half-second sampling:
  python benchmark/metrics_collector.py --duration 120 --interval 0.5 --output nvp_bench.csv

  # Monitor specific NIDS process:
  python benchmark/metrics_collector.py --pid 12345 --duration 60

  # Infinite collection (stop with Ctrl+C):
  python benchmark/metrics_collector.py --duration 0 --output live_metrics.csv
        """
    )
    parser.add_argument(
        "--duration", type=float, default=60,
        help="Collection duration in seconds (0 = infinite, default: 60)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Sampling interval in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark/results/metrics.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--label", type=str, default="system",
        help="Label for this run (e.g., 'NVP' or 'MVP')"
    )
    parser.add_argument(
        "--pid", type=int, default=None,
        help="Monitor a specific process by PID instead of system-wide"
    )

    args = parser.parse_args()

    if args.pid:
        collect_process_metrics(
            pid=args.pid,
            duration=args.duration,
            interval=args.interval,
            output_path=args.output,
            label=args.label,
        )
    else:
        collect_system_metrics(
            duration=args.duration,
            interval=args.interval,
            output_path=args.output,
            label=args.label,
        )


if __name__ == "__main__":
    main()
