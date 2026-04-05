import socket
import time
import sys

# ============================================================
# SAFETY FIRST: This script is for LOCAL NIDS TESTING ONLY.
# Do NOT use this script against networks you do not own.
# ============================================================

def simulate_udp_flood(target_ip="192.168.1.108", target_port=8000, duration=2):
    """
    Simulates a high-volume UDP flood attack to test NIDS detection.
    This targets feature distribution (Packet Counts, Small Payloads).
    """
    # Specifically designed payload (100 bytes) to target 
    # the 'NUM_PKTS_UP_TO_128_BYTES' feature in our model.
    PAYLOAD = b"X" * 100
    
    print("====================================================")
    print("   NIDS ATTACK SIMULATOR: LOCAL UDP FLOOD           ")
    print("====================================================")
    print(f"[*] DESTINATION: {target_ip}:{target_port}")
    print(f"[*] DURATION:    {duration} seconds")
    print(f"[*] PAYLOAD:     {len(PAYLOAD)} bytes per packet")
    print("[*] Starting flood in 1 second...")
    time.sleep(1)

    # Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    start_time = time.time()
    packet_count = 0
    total_bytes = 0

    print("[!] FLOODING IN PROGRESS...")
    
    try:
        while time.time() - start_time < duration:
            # Send the high-frequency packet
            sock.sendto(PAYLOAD, (target_ip, target_port))
            
            # Increment tallies
            packet_count += 1
            total_bytes += len(PAYLOAD)
            
            # Optional: Add micro-delay to prevent 100% CPU lock while still flooding
            # time.sleep(0.0001) 

        print("\n--- SIMULATION COMPLETE ---")
    except KeyboardInterrupt:
        print("\n[!] Simulation stopped by user.")
    except Exception as e:
        print(f"\n[✘] Error during simulation: {e}")
    finally:
        sock.close()

    # Reporting results
    print(f"[*] Summary:")
    print(f"    - Total Packets Sent: {packet_count}")
    print(f"    - Total Bytes Sent:   {total_bytes:_} bytes")
    print(f"    - Avg. Packet Rate:   {int(packet_count / duration)} packets/sec")
    print("[*] Check your NIDS Sniffer console for detection results.")

if __name__ == "__main__":
    simulate_udp_flood()
