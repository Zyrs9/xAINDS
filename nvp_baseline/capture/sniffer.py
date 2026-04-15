import threading
import time
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP
import queue

class LiveSniffer:
    """
    Captures live packets using Scapy and aggregates them into flows.
    Runs in a background thread to prevent blocking.
    """
    def __init__(self, interface=None, flow_timeout=5, max_packets_per_flow=50):
        self.interface = interface
        self.flow_timeout = flow_timeout
        self.max_packets_per_flow = max_packets_per_flow
        
        # flow_id: (src_ip, src_port, dst_ip, dst_port, proto)
        self.active_flows = defaultdict(list)
        self.flow_start_times = {}
        self.results_queue = queue.Queue(maxsize=100) # Store aggregated flow packet lists
        
        self.running = False
        self.sniffer_thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.sniffer_thread = threading.Thread(target=self._run_sniff, daemon=True)
        self.sniffer_thread.start()
        print(f"Background sniffer started on interface: {self.interface or 'Default'}")

    def stop(self):
        self.running = False
        if self.sniffer_thread:
            self.sniffer_thread.join(timeout=2)
        print("Background sniffer stopped.")

    def _get_flow_id(self, pkt):
        """Generates a unique 5-tuple identifier for the flow."""
        if IP in pkt:
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
                
            # Consistent flow ID for bidirectional mapping (sort ports/IPs)
            # For simplicity, we'll keep it one-directional
            return (src_ip, src_port, dst_ip, dst_port, proto)
        return None

    def _packet_handler(self, pkt):
        flow_id = self._get_flow_id(pkt)
        if not flow_id:
            return

        # Add packet to flow window
        self.active_flows[flow_id].append(pkt)
        
        if flow_id not in self.flow_start_times:
            self.flow_start_times[flow_id] = time.time()
            
        elapsed = time.time() - self.flow_start_times[flow_id]
        
        # Check if flow is ready for analysis (timeout or packet count)
        if len(self.active_flows[flow_id]) >= self.max_packets_per_flow or elapsed >= self.flow_timeout:
            # Move flow to results queue for backend
            flow_packets = self.active_flows.pop(flow_id)
            self.flow_start_times.pop(flow_id)
            
            if not self.results_queue.full():
                self.results_queue.put(flow_packets)

    def _run_sniff(self):
        try:
            # store=0 prevents Scapy from keeping all packets in memory
            sniff(
                iface=self.interface, 
                prn=self._packet_handler, 
                store=0, 
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"Sniffer Error: {e}")

    def get_next_flow(self, timeout=None):
        """Retrieves an aggregated flow (list of packets) from the queue."""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
