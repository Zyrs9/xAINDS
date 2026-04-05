import time
import numpy as np

class FlowFeatureExtractor:
    """
    Extracts behavioral flow-based features from a collection of packets belonging to a single flow.
    Flow is defined as (src_ip, src_port, dst_ip, dst_port, protocol).
    """
    
    @staticmethod
    def get_feature_names():
        """Returns the list of features in the order they are returned by extract()."""
        return ["packets_per_second", "avg_payload_size", "flow_duration", "byte_rate"]

    @staticmethod
    def extract(packets):
        """
        Extracts features from a list of scapy packets for a flow.
        Returns a numpy array of feature values.
        """
        if not packets:
            return np.zeros(len(FlowFeatureExtractor.get_feature_names()))

        num_packets = len(packets)
        start_time = packets[0].time
        end_time = packets[-1].time
        
        # Duration in seconds (ensure it's not zero to avoid div by zero)
        duration = float(end_time - start_time)
        safe_duration = max(duration, 0.001) 
        
        # Payload size (Raw Layer)
        total_payload_bytes = 0
        total_bytes = 0
        for pkt in packets:
            total_bytes += len(pkt)
            if pkt.haslayer("Raw"):
                total_payload_bytes += len(pkt["Raw"].load)
        
        # Features calculation
        packets_per_second = num_packets / safe_duration
        avg_payload_size = total_payload_bytes / num_packets
        byte_rate = total_bytes / safe_duration
        
        return np.array([
            packets_per_second,
            avg_payload_size,
            duration,
            byte_rate
        ])
