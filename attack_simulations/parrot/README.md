# Parrot Attack Simulation Scripts

These scripts generate bounded lab traffic from Parrot to the Debian victim network.

They are intended for the Hyper-V lab only:

- Attacker network: `10.10.10.0/24`
- Victim network: `10.10.20.0/24`
- Default target: `10.10.20.10`

The scripts refuse targets outside `10.10.20.0/24`.

## Setup on Parrot

Copy this folder to Parrot, then run:

```bash
cd attack_simulations/parrot
chmod +x *.sh
```

## Ubuntu Monitor

Run the NIDS monitor on Ubuntu before generating traffic:

```bash
cd /opt/nids-app/deployment
sudo .venv/bin/python tcpdump_monitor.py --interface eth1 --flow-timeout 3 --min-packets 10 --filter "host 10.10.20.10"
```

For HTTP/SYN demos:

```bash
sudo .venv/bin/python tcpdump_monitor.py --interface eth1 --flow-timeout 3 --min-packets 10 --filter "host 10.10.20.10 and tcp and port 80"
```

## Scenarios

### TCP SYN Burst

```bash
TARGET=10.10.20.10 PORT=80 COUNT=1000 ./syn_burst.sh
```

Common model-facing traits:

- Many small TCP packets.
- SYN flag concentration.
- High packets per second.

### UDP Burst

```bash
TARGET=10.10.20.10 PORT=53 COUNT=500 ./udp_burst.sh
```

Common traits:

- Stateless burst traffic.
- Elevated packet and byte rates.
- Often low response symmetry.

### ICMP Burst

```bash
TARGET=10.10.20.10 COUNT=500 ./icmp_burst.sh
```

Common traits:

- Repeated small ICMP packets.
- High packets per second.
- Low payload diversity.

### TCP Port Scan

```bash
TARGET=10.10.20.10 PORT_RANGE=1-1024 RATE=50 ./port_scan.sh
```

Common traits:

- Many short-lived flows.
- Many destination ports.
- Low byte count per flow.

### HTTP Request Burst

Start a simple HTTP server on Debian:

```bash
python3 -m http.server 8080
```

Then run from Parrot:

```bash
TARGET=10.10.20.10 PORT=8080 COUNT=200 ./http_burst.sh
```

Common traits:

- Repeated TCP application flows.
- Increased byte rate.
- Higher packet count than normal browsing.

## Demo Sequence

```bash
TARGET=10.10.20.10 ./run_demo_sequence.sh
```

Use this only after confirming the Ubuntu monitor and dashboard are running.
