import time
import requests
import json
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import warnings

# Suppress scapy warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
API_URL = "http://127.0.0.1:8000/analyze"
FLOW_PACKET_THRESHOLD = 20  # Process flow after 20 packets
FLOW_TIMEOUT_SECONDS = 5     # Or after 5 seconds of duration
INTERFACE = None             # None = default interface

# Global flow storage
# Key: bidirectional 5-tuple, Value: {packets: [], start_time: float, initiator: str}
active_flows = {}

def get_flow_key(pkt):
    """
    Generates a bidirectional flow key from the 5-tuple.
    Ensures that (SrcA, DstB, PortA, PortB) and (SrcB, DstA, PortB, PortA)
    return the same unique key string.
    """
    if not IP in pkt:
        return None
    
    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    proto = pkt[IP].proto
    
    src_port = 0
    dst_port = 0
    
    if TCP in pkt:
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
        
    # Sort IPs and Ports to make it bidirectional
    ips = sorted([src_ip, dst_ip])
    ports = sorted([src_port, dst_port])
    
    return f"{ips[0]}_{ips[1]}_{ports[0]}_{ports[1]}_{proto}"

def extract_and_send(flow_key):
    """
    Extracts the EXACT 10 behavioral features required for our NIDS Step 4.
    Sends the results to the FastAPI backend.
    """
    flow_data = active_flows[flow_key]
    packets = flow_data['packets']
    initiator_ip = flow_data['initiator']
    
    # Feature holders
    ttls = []
    pkt_lens = []
    ip_pkt_lens = []
    out_bytes = 0
    out_pkts = 0
    dst_to_src_bytes = 0
    pkts_small = 0
    
    for pkt in packets:
        raw_len = len(pkt)
        pkt_lens.append(raw_len)
        
        # NUM_PKTS_UP_TO_128_BYTES
        if raw_len <= 128:
            pkts_small += 1
            
        if IP in pkt:
            ttls.append(pkt[IP].ttl)
            ip_pkt_lens.append(pkt[IP].len)
            
            # Directional features
            if pkt[IP].src == initiator_ip:
                out_bytes += raw_len
                out_pkts += 1
            else:
                dst_to_src_bytes += raw_len

    # If IP layer was missing in all packets, set defaults
    if not ttls: ttls = [0]
    if not ip_pkt_lens: ip_pkt_lens = [0]
    
    # The EXACT 10 FEATURES
    features = {
        "MIN_TTL": float(min(ttls)),
        "MAX_TTL": float(max(ttls)),
        "SHORTEST_FLOW_PKT": float(min(pkt_lens)),
        "LONGEST_FLOW_PKT": float(max(pkt_lens)),
        "MIN_IP_PKT_LEN": float(min(ip_pkt_lens)),
        "MAX_IP_PKT_LEN": float(max(ip_pkt_lens)),
        "OUT_BYTES": float(out_bytes),
        "OUT_PKTS": float(out_pkts),
        "DST_TO_SRC_SECOND_BYTES": float(dst_to_src_bytes),
        "NUM_PKTS_UP_TO_128_BYTES": float(pkts_small)
    }

    # Clean up memory immediately after extraction
    del active_flows[flow_key]

    # Backend Communication
    try:
        response = requests.post(
            API_URL, 
            json={"features": features}, 
            timeout=3
        )
        if response.status_code == 200:
            res_json = response.json()
            status = res_json.get('status', 'Unknown')
            score = res_json.get('anomaly_score', 0.0)
            
            # Simple color coding in console
            color = "\033[91m" if status == "Anomaly" else "\033[92m"
            reset = "\033[0m"
            
            print(f"[{time.strftime('%H:%M:%S')}] {color}{status}{reset} | Score: {score:.4f} | Flow: {flow_key}")
        else:
            print(f"[✘] Backend error ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"[✘] Connection failed: Backend offline or unreachable.")

def packet_callback(pkt):
    """
    Scapy prn callback. Handles flow aggregation in real-time.
    """
    flow_key = get_flow_key(pkt)
    if not flow_key:
        return
    
    current_time = time.time()
    
    if flow_key not in active_flows:
        # Initialize a new flow
        active_flows[flow_key] = {
            'packets': [pkt],
            'start_time': current_time,
            'initiator': pkt[IP].src if IP in pkt else "Unknown"
        }
    else:
        # Add to existing flow
        active_flows[flow_key]['packets'].append(pkt)
        
        # Check thresholds for processing
        flow_duration = current_time - active_flows[flow_key]['start_time']
        packet_count = len(active_flows[flow_key]['packets'])
        
        if packet_count >= FLOW_PACKET_THRESHOLD or flow_duration >= FLOW_TIMEOUT_SECONDS:
            extract_and_send(flow_key)

def start_sniffer():
    print("====================================================")
    print("   NIDS REAL-TIME SNIFFER & FLOW AGGREGATOR         ")
    print("====================================================")
    print(f"[*] Targeting Backend: {API_URL}")
    print(f"[*] Sniffing on: {INTERFACE or 'All Interfaces'}")
    print("[*] Press Ctrl+C to stop.")
    
    try:
        # Use store=0 to prevent Scapy from storing packets in global list (memory leak)
        sniff(iface=INTERFACE, prn=packet_callback, store=0)
    except Exception as e:
        print(f"[✘] Sniffer error: {e}")
        print("[!] Ensure you are running as Administrator (sudo/root)")

if __name__ == "__main__":
    start_sniffer()
