# flow_engine.py (BIDIRECTIONAL VERSION) -- UPDATED FOR YOUR TOP-20

import time
import threading
import ipaddress
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
from scapy.layers.inet import IP, TCP, UDP

FlowKey = Tuple[str, int, str, int, str]  # (ip_a, port_a, ip_b, port_b, proto)


@dataclass
class FlowStats:
    flow_key: FlowKey
    start_time: float
    last_time: float

    # forward direction = direction of first packet
    fwd_src_ip: str
    fwd_src_port: int
    dst_port: int = 0  # destination port of the first packet

    # counters
    total_packets: int = 0
    total_bytes: int = 0
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    # packet lengths
    all_lens: List[int] = field(default_factory=list)
    fwd_lens: List[int] = field(default_factory=list)
    bwd_lens: List[int] = field(default_factory=list)

    # timestamps
    all_times: List[float] = field(default_factory=list)

    # init TCP window
    init_win_bytes_forward: int = 0
    init_win_bytes_backward: int = 0
    _got_fwd_window: bool = False
    _got_bwd_window: bool = False

    # header length totals
    fwd_header_len: int = 0
    bwd_header_len: int = 0


def _proto_of(pkt) -> Optional[str]:
    if pkt.haslayer(TCP):
        return "TCP"
    if pkt.haslayer(UDP):
        return "UDP"
    return None


def _ports_of(pkt) -> Tuple[int, int]:
    if pkt.haslayer(TCP):
        return int(pkt[TCP].sport), int(pkt[TCP].dport)
    if pkt.haslayer(UDP):
        return int(pkt[UDP].sport), int(pkt[UDP].dport)
    return 0, 0


def _pkt_len(pkt) -> int:
    if pkt.haslayer(IP) and hasattr(pkt[IP], "len") and pkt[IP].len is not None:
        return int(pkt[IP].len)
    return int(len(pkt))


def _ip_to_int(ip_str: str) -> int:
    return int(ipaddress.ip_address(ip_str))


def _canonical_flow_key(src_ip: str, sport: int, dst_ip: str, dport: int, proto: str) -> FlowKey:
    a = (_ip_to_int(src_ip), sport, src_ip)
    b = (_ip_to_int(dst_ip), dport, dst_ip)
    if (a[0], a[1]) <= (b[0], b[1]):
        return (a[2], a[1], b[2], b[1], proto)
    else:
        return (b[2], b[1], a[2], a[1], proto)


def _header_len_bytes(pkt) -> int:
    """
    Approx header length for CIC-style "Header Length" features:
    IP header + TCP/UDP header.
    """
    ip_h = 0
    if pkt.haslayer(IP) and hasattr(pkt[IP], "ihl") and pkt[IP].ihl is not None:
        ip_h = int(pkt[IP].ihl) * 4
    else:
        ip_h = 20  # fallback typical IPv4 header

    if pkt.haslayer(TCP):
        # TCP data offset * 4
        if hasattr(pkt[TCP], "dataofs") and pkt[TCP].dataofs is not None:
            return ip_h + int(pkt[TCP].dataofs) * 4
        return ip_h + 20  # fallback TCP header
    if pkt.haslayer(UDP):
        return ip_h + 8   # UDP header fixed
    return ip_h


class FlowEngine:
    def __init__(self, flow_timeout_sec: float = 5.0, inactive_timeout_sec: float = 3.0):
        self.flow_timeout_sec = flow_timeout_sec
        self.inactive_timeout_sec = inactive_timeout_sec
        self._lock = threading.Lock()
        self.flows: Dict[FlowKey, FlowStats] = {}

    def process_packet(self, pkt) -> None:
        if not pkt.haslayer(IP):
            return

        proto = _proto_of(pkt)
        if proto is None:
            return

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        sport, dport = _ports_of(pkt)
        if sport == 0 and dport == 0:
            return

        now = float(getattr(pkt, "time", time.time()))
        plen = _pkt_len(pkt)
        hlen = _header_len_bytes(pkt)

        key: FlowKey = _canonical_flow_key(src_ip, sport, dst_ip, dport, proto)

        with self._lock:
            fs = self.flows.get(key)
            if fs is None:
                fs = FlowStats(
                    flow_key=key,
                    start_time=now,
                    last_time=now,
                    fwd_src_ip=src_ip,
                    fwd_src_port=sport,
                    dst_port=dport,
                )
                self.flows[key] = fs

            fs.last_time = now

            # direction = match first packet's src endpoint
            is_fwd = (src_ip == fs.fwd_src_ip and sport == fs.fwd_src_port)

            fs.total_packets += 1
            fs.total_bytes += plen
            fs.all_lens.append(plen)
            fs.all_times.append(now)

            if is_fwd:
                fs.fwd_packets += 1
                fs.fwd_bytes += plen
                fs.fwd_lens.append(plen)
                fs.fwd_header_len += hlen

                # Init_Win_bytes_forward: first forward TCP window
                if (not fs._got_fwd_window) and pkt.haslayer(TCP):
                    fs.init_win_bytes_forward = int(pkt[TCP].window)
                    fs._got_fwd_window = True
            else:
                fs.bwd_packets += 1
                fs.bwd_bytes += plen
                fs.bwd_lens.append(plen)
                fs.bwd_header_len += hlen

                # Init_Win_bytes_backward: first backward TCP window
                if (not fs._got_bwd_window) and pkt.haslayer(TCP):
                    fs.init_win_bytes_backward = int(pkt[TCP].window)
                    fs._got_bwd_window = True

    def expire_flows(self) -> List[FlowStats]:
        now = time.time()
        expired: List[FlowStats] = []

        with self._lock:
            keys_to_delete = []
            for k, fs in self.flows.items():
                lifetime = fs.last_time - fs.start_time
                inactive = now - fs.last_time
                if lifetime >= self.flow_timeout_sec or inactive >= self.inactive_timeout_sec:
                    expired.append(fs)
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del self.flows[k]

        return expired

    @staticmethod
    def extract_top20_features(fs: FlowStats) -> Dict[str, float]:
        # arrays
        lens_arr = np.array(fs.all_lens, dtype=float) if fs.all_lens else np.array([0.0], dtype=float)
        fwd_arr  = np.array(fs.fwd_lens, dtype=float) if fs.fwd_lens else np.array([0.0], dtype=float)
        bwd_arr  = np.array(fs.bwd_lens, dtype=float) if fs.bwd_lens else np.array([0.0], dtype=float)

        # min/max safe
        min_len = float(np.min(lens_arr)) if lens_arr.size else 0.0
        max_len = float(np.max(lens_arr)) if lens_arr.size else 0.0

        # average packet size (CIC-style)
        avg_pkt_size = float(fs.total_bytes / fs.total_packets) if fs.total_packets > 0 else 0.0

        # IMPORTANT: output EXACT feature names your model expects
        feats = {
            "Init_Win_bytes_forward": float(fs.init_win_bytes_forward),
            "Destination Port": float(fs.dst_port),

            "Fwd Packet Length Mean": float(np.mean(fwd_arr)),
            "Min Packet Length": float(min_len),
            "Max Packet Length": float(max_len),

            "Init_Win_bytes_backward": float(fs.init_win_bytes_backward),
            "Bwd Packet Length Min": float(np.min(bwd_arr)),

            "Average Packet Size": float(avg_pkt_size),
            "Packet Length Std": float(np.std(lens_arr)),
            "Subflow Fwd Bytes": float(fs.fwd_bytes),
            "Packet Length Variance": float(np.var(lens_arr)),

            "Avg Bwd Segment Size": float(np.mean(bwd_arr)),
            "Avg Fwd Segment Size": float(np.mean(fwd_arr)),

            "Bwd Packet Length Max": float(np.max(bwd_arr)),
            "Fwd Packet Length Max": float(np.max(fwd_arr)),

            # your dataset has this duplicate column name; we mirror it
            "Fwd Header Length.1": float(fs.fwd_header_len),

            "Bwd Packet Length Std": float(np.std(bwd_arr)),
            "Fwd Packet Length Min": float(np.min(fwd_arr)),

            "Fwd Header Length": float(fs.fwd_header_len),
            "Bwd Header Length": float(fs.bwd_header_len),
        }
        return feats
