/*
 * eBPF/XDP Flow Tracker — Kernel-Space Packet Parsing & Flow Aggregation
 *
 * This program hooks at the XDP (eXpress Data Path) layer — the earliest
 * possible packet interception point in the Linux kernel, BEFORE sk_buff
 * allocation. This eliminates the user-space copy overhead that cripples
 * Scapy-based NIDS implementations.
 *
 * Architecture:
 *   1. XDP hook parses Ethernet → IP → TCP/UDP headers
 *   2. Extracts 5-tuple flow key (src_ip, dst_ip, src_port, dst_port, proto)
 *   3. Maintains per-flow statistics in a BPF Hash Map
 *   4. When flows hit packet threshold or timeout, exports stats to user-space
 *      via a Perf Event Array
 *
 * The 10 features extracted here match the UNSW-NB15 feature set exactly:
 *   MIN_TTL, MAX_TTL, SHORTEST_FLOW_PKT, LONGEST_FLOW_PKT,
 *   MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, OUT_BYTES, OUT_PKTS,
 *   DST_TO_SRC_SECOND_BYTES, NUM_PKTS_UP_TO_128_BYTES
 *
 * Compile (standalone):
 *   clang -O2 -target bpf -c flow_tracker.c -o flow_tracker.o
 *
 * Note: When used with BCC (Python), BCC compiles this inline — no manual
 *       compilation needed. The standalone compile is for libbpf deployments.
 *
 * Author: NIDS Research Team
 * License: GPL-2.0 (required for eBPF programs)
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Constants
 * ───────────────────────────────────────────────────────────────────────── */

#define MAX_FLOWS          65536   /* Maximum concurrent tracked flows      */
#define FLOW_PKT_THRESHOLD 20     /* Export flow after N packets            */
#define FLOW_TIMEOUT_NS    5000000000ULL  /* 5 seconds in nanoseconds      */
#define SMALL_PKT_CUTOFF   128    /* Packets <= this are "small"            */

/* ─────────────────────────────────────────────────────────────────────────
 * Data Structures
 * ───────────────────────────────────────────────────────────────────────── */

/*
 * 5-tuple flow key for bidirectional flow identification.
 * IPs and ports are sorted to ensure bidirectional matching:
 *   (A→B) and (B→A) produce the same key.
 */
struct flow_key {
    __u32 ip_lo;       /* Lower IP address (network byte order)   */
    __u32 ip_hi;       /* Higher IP address (network byte order)  */
    __u16 port_lo;     /* Lower port number                       */
    __u16 port_hi;     /* Higher port number                      */
    __u8  protocol;    /* IP protocol (TCP=6, UDP=17)             */
    __u8  pad[3];      /* Alignment padding                       */
};

/*
 * Per-flow state maintained in the BPF hash map.
 * These fields directly compute our 10 ML features.
 */
struct flow_state {
    /* Timing */
    __u64 first_seen_ns;       /* Timestamp of first packet (ktime)       */
    __u64 last_seen_ns;        /* Timestamp of most recent packet         */

    /* Feature: MIN_TTL, MAX_TTL */
    __u8  min_ttl;
    __u8  max_ttl;

    /* Feature: SHORTEST_FLOW_PKT, LONGEST_FLOW_PKT (raw Ethernet frame) */
    __u16 min_pkt_len;
    __u16 max_pkt_len;

    /* Feature: MIN_IP_PKT_LEN, MAX_IP_PKT_LEN (IP total length) */
    __u16 min_ip_pkt_len;
    __u16 max_ip_pkt_len;

    /* Feature: OUT_BYTES, OUT_PKTS (initiator → destination) */
    __u64 out_bytes;
    __u32 out_pkts;

    /* Feature: DST_TO_SRC_SECOND_BYTES (reverse direction) */
    __u64 dst_to_src_bytes;

    /* Feature: NUM_PKTS_UP_TO_128_BYTES */
    __u32 pkts_up_to_128;

    /* Bookkeeping */
    __u32 total_pkts;          /* Total packet count in this flow         */
    __u32 initiator_ip;        /* IP of the flow initiator (first pkt src)*/
};

/*
 * Exported flow record — sent to user-space via perf event.
 * Contains the final computed 10-feature vector.
 */
struct flow_export {
    /* Flow identification */
    struct flow_key key;

    /* The 10 ML features (matching UNSW-NB15 column order) */
    __u32 min_ttl;                   /* 1. MIN_TTL                     */
    __u32 max_ttl;                   /* 2. MAX_TTL                     */
    __u32 shortest_flow_pkt;         /* 3. SHORTEST_FLOW_PKT           */
    __u32 longest_flow_pkt;          /* 4. LONGEST_FLOW_PKT            */
    __u32 min_ip_pkt_len;            /* 5. MIN_IP_PKT_LEN              */
    __u32 max_ip_pkt_len;            /* 6. MAX_IP_PKT_LEN              */
    __u64 out_bytes;                 /* 7. OUT_BYTES                   */
    __u32 out_pkts;                  /* 8. OUT_PKTS                    */
    __u64 dst_to_src_second_bytes;   /* 9. DST_TO_SRC_SECOND_BYTES     */
    __u32 num_pkts_up_to_128;        /* 10. NUM_PKTS_UP_TO_128_BYTES   */

    /* Metadata */
    __u32 total_pkts;
    __u64 flow_duration_ns;
};

/* ─────────────────────────────────────────────────────────────────────────
 * BPF Maps
 * ───────────────────────────────────────────────────────────────────────── */

/* Hash map: flow_key → flow_state */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_FLOWS);
    __type(key, struct flow_key);
    __type(value, struct flow_state);
} flow_table SEC(".maps");

/* Perf event array for exporting completed flows to user-space */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} flow_events SEC(".maps");

/* ─────────────────────────────────────────────────────────────────────────
 * Helper Functions
 * ───────────────────────────────────────────────────────────────────────── */

/*
 * Construct a bidirectional flow key by sorting IPs and ports.
 * This ensures (A→B) and (B→A) map to the same flow entry.
 */
static __always_inline void
make_flow_key(struct flow_key *fk, __u32 src_ip, __u32 dst_ip,
              __u16 src_port, __u16 dst_port, __u8 proto)
{
    if (src_ip < dst_ip || (src_ip == dst_ip && src_port <= dst_port)) {
        fk->ip_lo   = src_ip;
        fk->ip_hi   = dst_ip;
        fk->port_lo = src_port;
        fk->port_hi = dst_port;
    } else {
        fk->ip_lo   = dst_ip;
        fk->ip_hi   = src_ip;
        fk->port_lo = dst_port;
        fk->port_hi = src_port;
    }
    fk->protocol = proto;
    fk->pad[0] = fk->pad[1] = fk->pad[2] = 0;
}

/*
 * Export a completed flow's statistics to user-space and delete from map.
 */
static __always_inline void
export_flow(void *ctx, struct flow_key *fk, struct flow_state *fs)
{
    struct flow_export evt = {};

    /* Copy flow key */
    __builtin_memcpy(&evt.key, fk, sizeof(struct flow_key));

    /* Map internal state → 10-feature export format */
    evt.min_ttl              = (__u32)fs->min_ttl;
    evt.max_ttl              = (__u32)fs->max_ttl;
    evt.shortest_flow_pkt    = (__u32)fs->min_pkt_len;
    evt.longest_flow_pkt     = (__u32)fs->max_pkt_len;
    evt.min_ip_pkt_len       = (__u32)fs->min_ip_pkt_len;
    evt.max_ip_pkt_len       = (__u32)fs->max_ip_pkt_len;
    evt.out_bytes            = fs->out_bytes;
    evt.out_pkts             = fs->out_pkts;
    evt.dst_to_src_second_bytes = fs->dst_to_src_bytes;
    evt.num_pkts_up_to_128   = fs->pkts_up_to_128;

    /* Metadata */
    evt.total_pkts      = fs->total_pkts;
    evt.flow_duration_ns = fs->last_seen_ns - fs->first_seen_ns;

    /* Submit to user-space via perf event ring buffer */
    bpf_perf_event_output(ctx, &flow_events, BPF_F_CURRENT_CPU,
                          &evt, sizeof(evt));

    /* Remove the flow from the hash table */
    bpf_map_delete_elem(&flow_table, fk);
}

/* ─────────────────────────────────────────────────────────────────────────
 * XDP Program Entry Point
 * ───────────────────────────────────────────────────────────────────────── */

SEC("xdp")
int xdp_flow_tracker(struct xdp_md *ctx)
{
    /* Packet boundaries */
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    /* ── Layer 2: Ethernet ────────────────────────────────────────────── */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    /* Only process IPv4 */
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    /* ── Layer 3: IP ──────────────────────────────────────────────────── */
    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end)
        return XDP_PASS;

    /* Validate IP header length */
    __u32 ip_hdr_len = iph->ihl * 4;
    if (ip_hdr_len < sizeof(struct iphdr))
        return XDP_PASS;
    if ((void *)iph + ip_hdr_len > data_end)
        return XDP_PASS;

    __u32 src_ip   = iph->saddr;
    __u32 dst_ip   = iph->daddr;
    __u8  proto    = iph->protocol;
    __u8  ttl      = iph->ttl;
    __u16 ip_len   = bpf_ntohs(iph->tot_len);

    /* Total frame length (L2 + L3 + L4 + payload) */
    __u32 pkt_len  = (__u32)(data_end - data);

    /* ── Layer 4: TCP / UDP port extraction ───────────────────────────── */
    __u16 src_port = 0;
    __u16 dst_port = 0;

    if (proto == IPPROTO_TCP) {
        struct tcphdr *tcph = (void *)iph + ip_hdr_len;
        if ((void *)(tcph + 1) > data_end)
            return XDP_PASS;
        src_port = bpf_ntohs(tcph->source);
        dst_port = bpf_ntohs(tcph->dest);
    } else if (proto == IPPROTO_UDP) {
        struct udphdr *udph = (void *)iph + ip_hdr_len;
        if ((void *)(udph + 1) > data_end)
            return XDP_PASS;
        src_port = bpf_ntohs(udph->source);
        dst_port = bpf_ntohs(udph->dest);
    } else {
        /* Skip non-TCP/UDP traffic (ICMP, etc.) */
        return XDP_PASS;
    }

    /* ── Build bidirectional flow key ─────────────────────────────────── */
    struct flow_key fk = {};
    make_flow_key(&fk, src_ip, dst_ip, src_port, dst_port, proto);

    __u64 now = bpf_ktime_get_ns();

    /* ── Flow state lookup / creation ─────────────────────────────────── */
    struct flow_state *fs = bpf_map_lookup_elem(&flow_table, &fk);

    if (!fs) {
        /* New flow: initialize state */
        struct flow_state new_fs = {};

        new_fs.first_seen_ns  = now;
        new_fs.last_seen_ns   = now;
        new_fs.min_ttl        = ttl;
        new_fs.max_ttl        = ttl;
        new_fs.min_pkt_len    = (__u16)pkt_len;
        new_fs.max_pkt_len    = (__u16)pkt_len;
        new_fs.min_ip_pkt_len = ip_len;
        new_fs.max_ip_pkt_len = ip_len;
        new_fs.total_pkts     = 1;
        new_fs.initiator_ip   = src_ip;

        /* First packet is always "outbound" (initiator → destination) */
        new_fs.out_bytes = pkt_len;
        new_fs.out_pkts  = 1;
        new_fs.dst_to_src_bytes = 0;

        /* Small packet counter */
        if (pkt_len <= SMALL_PKT_CUTOFF)
            new_fs.pkts_up_to_128 = 1;

        bpf_map_update_elem(&flow_table, &fk, &new_fs, BPF_ANY);
        return XDP_PASS;
    }

    /* ── Update existing flow state ───────────────────────────────────── */

    fs->last_seen_ns = now;
    fs->total_pkts  += 1;

    /* TTL min/max */
    if (ttl < fs->min_ttl) fs->min_ttl = ttl;
    if (ttl > fs->max_ttl) fs->max_ttl = ttl;

    /* Packet length min/max (raw frame) */
    if ((__u16)pkt_len < fs->min_pkt_len) fs->min_pkt_len = (__u16)pkt_len;
    if ((__u16)pkt_len > fs->max_pkt_len) fs->max_pkt_len = (__u16)pkt_len;

    /* IP packet length min/max */
    if (ip_len < fs->min_ip_pkt_len) fs->min_ip_pkt_len = ip_len;
    if (ip_len > fs->max_ip_pkt_len) fs->max_ip_pkt_len = ip_len;

    /* Directional byte/packet counters */
    if (src_ip == fs->initiator_ip) {
        fs->out_bytes += pkt_len;
        fs->out_pkts  += 1;
    } else {
        fs->dst_to_src_bytes += pkt_len;
    }

    /* Small packet counter */
    if (pkt_len <= SMALL_PKT_CUTOFF)
        fs->pkts_up_to_128 += 1;

    /* ── Check export conditions ──────────────────────────────────────── */
    __u64 duration = now - fs->first_seen_ns;

    if (fs->total_pkts >= FLOW_PKT_THRESHOLD ||
        duration >= FLOW_TIMEOUT_NS) {
        export_flow(ctx, &fk, fs);
    }

    /* XDP_PASS: Let the packet continue through the network stack.
     * We are a PASSIVE monitor — never drop or modify traffic. */
    return XDP_PASS;
}

/* GPL license is REQUIRED for eBPF programs using kernel helpers */
char _license[] SEC("license") = "GPL";
