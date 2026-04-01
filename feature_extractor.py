"""
╔══════════════════════════════════════════════════════════════════╗
║    FEATURE_EXTRACTOR.PY - TRÍCH XUẤT FEATURES TỪ PACKETS      ║
╚══════════════════════════════════════════════════════════════════╝

LÝ THUYẾT: Feature Extraction cho Network IDS
─────────────────────────────────────────────

Packet đến dạng raw bytes → cần trích xuất features có ý nghĩa.

Có 2 cách tiếp cận:

1. PACKET-BASED (Per-packet features):
   Trích xuất từ từng packet riêng lẻ:
   - Protocol (TCP/UDP/ICMP)
   - Source/Destination Port
   - Packet size
   - TCP Flags
   - TTL
   → Nhanh, nhưng mỗi packet thiếu context

2. FLOW-BASED (Aggregate features):
   Nhóm packets theo 5-tuple, tính statistics:
   - Duration of flow
   - Total bytes, packets
   - Mean, Std of packet sizes
   - Flag counts
   - Inter-arrival time statistics
   → Chính xác hơn, standard cho IDS hiện đại

Project này implement CẢ HAI:
- Packet-based cho real-time (latency thấp)
- Flow-based cho accuracy cao (aggregate → predict)
"""

import time
import numpy as np
from collections import defaultdict

import config


class PacketFeatureExtractor:
    """
    Trích xuất features từ individual packets.

    LÝ THUYẾT: TCP/IP Header Fields
    ────────────────────────────────
    IP Header:
      - Protocol: 6=TCP, 17=UDP, 1=ICMP
      - TTL: Time To Live (hop count)
      - Total Length: Size of packet

    TCP Header:
      - Source Port, Destination Port
      - Flags: SYN, ACK, FIN, RST, PSH, URG
      - Window Size: Flow control
      - Sequence Number: Ordering

    UDP Header:
      - Source Port, Destination Port
      - Length
    """

    def __init__(self):
        # 18 features khớp với CANONICAL_FEATURES trong dataset_loader.py
        # 'Protocol' bị loại vì CICIDS2017 không có cột này
        self.feature_names = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Down/Up Ratio',
            'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
            'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count',
            'Active Mean', 'Idle Mean',
            'Flow IAT Mean', 'Flow IAT Std',
            'Destination Port',
        ]

    def extract_from_packet_dict(self, packet_info):
        """
        Trích xuất features từ packet info dictionary.

        Input: Dictionary với các trường từ parsed packet
        Output: numpy array 18 features (matching CANONICAL_FEATURES in dataset_loader)

        LÝ THUYẾT: Feature Alignment
        ────────────────────────────
        Features PHẢI khớp chính xác với features lúc TRAIN:
        - Cùng số lượng features
        - Cùng thứ tự features
        - Cùng scale (dùng cùng scaler)
        Nếu không → model predict sai hoàn toàn!
        """
        features = np.zeros(len(self.feature_names))

        features[0] = packet_info.get('duration', 0)
        features[1] = packet_info.get('fwd_packets', 1)
        features[2] = packet_info.get('bwd_packets', 0)
        features[3] = packet_info.get('flow_bytes_per_sec', 0)
        features[4] = packet_info.get('flow_packets_per_sec', 0)
        features[5] = packet_info.get('down_up_ratio', 1.0)
        features[6] = packet_info.get('fwd_pkt_len_mean', 0)
        features[7] = packet_info.get('bwd_pkt_len_mean', 0)
        features[8] = packet_info.get('syn_count', 0)
        features[9] = packet_info.get('fin_count', 0)
        features[10] = packet_info.get('rst_count', 0)
        features[11] = packet_info.get('psh_count', 0)
        features[12] = packet_info.get('ack_count', 0)
        features[13] = packet_info.get('active_mean', 0)
        features[14] = packet_info.get('idle_mean', 0)
        features[15] = packet_info.get('iat_mean', 0)
        features[16] = packet_info.get('iat_std', 0)
        features[17] = packet_info.get('dst_port', 0)

        return features


class FlowAggregator:
    """
    ╔════════════════════════════════════════════════════════╗
    ║  FLOW AGGREGATOR - Nhóm packets thành flows          ║
    ╠════════════════════════════════════════════════════════╣
    ║  Flow = nhóm packets cùng 5-tuple:                   ║
    ║  (src_ip, dst_ip, src_port, dst_port, protocol)      ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Flow-based Analysis
    ──────────────────────────────
    Network traffic tự nhiên tổ chức theo FLOWS:
    - HTTP request/response: 1 flow
    - SSH session: 1 flow (dài)
    - DNS query: 1 flow (ngắn)

    Tại sao flow-based tốt hơn packet-based?
    1. Bắt được patterns: DDoS = nhiều flows ngắn tới cùng 1 server
    2. Bắt được temporal features: duration, packet rate
    3. Giảm noise: 1 flow = 1 data point (thay vì hàng nghìn packets)

    TIMEOUT MECHANISM:
    - Flow kết thúc khi: FIN/RST flag, hoặc không có packet mới sau X giây
    - Thường timeout = 60s-120s cho active flows
    """

    def __init__(self, flow_timeout=60.0):
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'last_time': None,
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': [],
            'bwd_bytes': [],
            'flags': defaultdict(int),
            'iat_list': [],
        })
        self.flow_timeout = flow_timeout
        self.extractor = PacketFeatureExtractor()

    def _flow_key(self, packet_info):
        """Tạo flow key từ 5-tuple."""
        return (
            packet_info.get('src_ip', '0.0.0.0'),
            packet_info.get('dst_ip', '0.0.0.0'),
            packet_info.get('src_port', 0),
            packet_info.get('dst_port', 0),
            packet_info.get('protocol', 0),
        )

    def add_packet(self, packet_info):
        """
        Thêm packet vào flow tương ứng.
        Trả về flow features nếu flow kết thúc, None nếu chưa.
        """
        key = self._flow_key(packet_info)
        flow = self.flows[key]
        now = packet_info.get('timestamp', time.time())

        if flow['start_time'] is None:
            flow['start_time'] = now

        # Tính IAT (Inter-Arrival Time)
        if flow['last_time'] is not None:
            iat = (now - flow['last_time']) * 1e6  # Convert to microseconds
            flow['iat_list'].append(iat)

        flow['last_time'] = now

        # Forward vs Backward packets
        is_forward = True  # Simplified: first direction = forward
        if flow['fwd_packets'] + flow['bwd_packets'] > 0:
            is_forward = (packet_info.get('src_port', 0) ==
                         key[2])  # Same as original src_port

        pkt_size = packet_info.get('size', 0)
        if is_forward:
            flow['fwd_packets'] += 1
            flow['fwd_bytes'].append(pkt_size)
        else:
            flow['bwd_packets'] += 1
            flow['bwd_bytes'].append(pkt_size)

        # TCP Flags
        for flag in ['syn', 'ack', 'fin', 'rst', 'psh']:
            if packet_info.get(f'{flag}_flag', False):
                flow['flags'][flag] += 1

        # Check if flow should be exported (FIN, RST, or timeout)
        if (packet_info.get('fin_flag', False) or
            packet_info.get('rst_flag', False)):
            return self._export_flow(key)

        return None

    def check_timeouts(self):
        """Kiểm tra và export các flows đã timeout."""
        now = time.time()
        expired_flows = []

        for key, flow in self.flows.items():
            if flow['last_time'] and (now - flow['last_time']) > self.flow_timeout:
                expired_flows.append(key)

        results = []
        for key in expired_flows:
            features = self._export_flow(key)
            if features is not None:
                results.append(features)

        return results

    def _export_flow(self, key):
        """
        Tính toán features từ flow data và xóa flow.

        LÝ THUYẾT: Statistical Features
        ────────────────────────────────
        Từ raw packet data trong 1 flow, tính:
        - Duration = last_time - start_time
        - Flow Bytes/s = total_bytes / duration
        - Packet Length Mean = mean(all_packet_sizes)
        - IAT Mean = mean(inter_arrival_times)
        - IAT Std = std(inter_arrival_times)
        - Flag Counts = count of each TCP flag type
        """
        flow = self.flows.pop(key, None)
        if flow is None or flow['start_time'] is None:
            return None

        duration = (flow['last_time'] - flow['start_time']) * 1e6  # microseconds
        total_packets = flow['fwd_packets'] + flow['bwd_packets']
        fwd_bytes = flow['fwd_bytes']
        bwd_bytes = flow['bwd_bytes']
        total_bytes = sum(fwd_bytes) + sum(bwd_bytes)

        duration_sec = max(duration / 1e6, 1e-6)  # Avoid division by zero

        packet_info = {
            'duration': duration,
            'fwd_packets': flow['fwd_packets'],
            'bwd_packets': flow['bwd_packets'],
            'flow_bytes_per_sec': total_bytes / duration_sec,
            'flow_packets_per_sec': total_packets / duration_sec,
            'down_up_ratio': (sum(bwd_bytes) / max(sum(fwd_bytes), 1)),
            'fwd_pkt_len_mean': np.mean(fwd_bytes) if fwd_bytes else 0,
            'bwd_pkt_len_mean': np.mean(bwd_bytes) if bwd_bytes else 0,
            'syn_count': flow['flags']['syn'],
            'fin_count': flow['flags']['fin'],
            'rst_count': flow['flags']['rst'],
            'psh_count': flow['flags']['psh'],
            'ack_count': flow['flags']['ack'],
            'active_mean': duration / max(total_packets, 1),
            'idle_mean': np.mean(flow['iat_list']) if flow['iat_list'] else 0,
            'iat_mean': np.mean(flow['iat_list']) if flow['iat_list'] else 0,
            'iat_std': np.std(flow['iat_list']) if len(flow['iat_list']) > 1 else 0,
            'protocol': key[4],
            'dst_port': key[3],
        }

        return self.extractor.extract_from_packet_dict(packet_info)


def generate_simulated_packet():
    """
    Tạo simulated packet cho testing/benchmark.

    LÝ THUYẾT: Packet Simulation
    ────────────────────────────
    Trong môi trường dev/test, không phải lúc nào cũng có
    live traffic để capture. Simulated packets giúp:
    1. Test pipeline end-to-end
    2. Benchmark performance
    3. Reproducible experiments
    """
    is_attack = np.random.random() < 0.2  # 20% attack

    if is_attack:
        return {
            'src_ip': f'10.0.{np.random.randint(1,5)}.{np.random.randint(1,255)}',
            'dst_ip': '192.168.1.100',
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([22, 23, 80, 443, 445, 3389]),
            'protocol': np.random.choice([6, 17, 1]),
            'size': np.random.randint(40, 200),
            'timestamp': time.time(),
            'syn_flag': np.random.random() < 0.6,
            'ack_flag': np.random.random() < 0.3,
            'fin_flag': False,
            'rst_flag': np.random.random() < 0.3,
            'psh_flag': np.random.random() < 0.4,
            'is_attack': True,
        }
    else:
        return {
            'src_ip': '192.168.1.100',
            'dst_ip': f'10.0.0.{np.random.randint(1,255)}',
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 53, 8080]),
            'protocol': np.random.choice([6, 17], p=[0.7, 0.3]),
            'size': np.random.randint(40, 1500),
            'timestamp': time.time(),
            'syn_flag': np.random.random() < 0.1,
            'ack_flag': np.random.random() < 0.7,
            'fin_flag': np.random.random() < 0.1,
            'rst_flag': False,
            'psh_flag': np.random.random() < 0.3,
            'is_attack': False,
        }


def generate_simulated_features():
    """
    Tạo simulated feature vector (đã extract) cho benchmark AI inference.
    Nhanh hơn generate_simulated_packet vì skip extraction step.
    """
    is_attack = np.random.random() < 0.2

    if is_attack:
        return np.array([
            np.random.exponential(1000000),         # Flow Duration
            np.random.poisson(50),                   # Total Fwd Packets
            np.random.poisson(3),                    # Total Backward Packets
            np.random.lognormal(10, 3),              # Flow Bytes/s
            np.random.lognormal(5, 2),               # Flow Packets/s
            np.random.uniform(0.01, 0.5),            # Down/Up Ratio
            np.random.normal(100, 50),               # Fwd Packet Length Mean
            np.random.normal(50, 30),                # Bwd Packet Length Mean
            np.random.choice([0, 1, 2, 5, 10]),      # SYN Flag Count
            np.random.choice([0, 1]),                # FIN Flag Count
            np.random.choice([0, 1, 2]),             # RST Flag Count
            np.random.poisson(8),                    # PSH Flag Count
            np.random.poisson(15),                   # ACK Flag Count
            np.random.exponential(500),              # Active Mean
            np.random.exponential(200),              # Idle Mean
            np.random.exponential(10000),            # Flow IAT Mean
            np.random.exponential(5000),             # Flow IAT Std
            np.random.choice([22, 23, 80, 443]),     # Destination Port
        ], dtype=np.float64)
    else:
        return np.array([
            np.random.exponential(500000),           # Flow Duration
            np.random.poisson(10),                   # Total Fwd Packets
            np.random.poisson(8),                    # Total Backward Packets
            np.random.lognormal(8, 2),               # Flow Bytes/s
            np.random.lognormal(3, 1),               # Flow Packets/s
            np.random.uniform(0.5, 5.0),             # Down/Up Ratio
            np.random.normal(200, 100),              # Fwd Packet Length Mean
            np.random.normal(500, 200),              # Bwd Packet Length Mean
            np.random.choice([0, 1]),                # SYN Flag Count
            np.random.choice([0, 1]),                # FIN Flag Count
            np.random.choice([0, 1]),                # RST Flag Count
            np.random.poisson(2),                    # PSH Flag Count
            np.random.poisson(5),                    # ACK Flag Count
            np.random.exponential(100),              # Active Mean
            np.random.exponential(1000),             # Idle Mean
            np.random.exponential(50000),            # Flow IAT Mean
            np.random.exponential(30000),            # Flow IAT Std
            np.random.choice([80, 443, 53, 8080]),   # Destination Port
        ], dtype=np.float64)
