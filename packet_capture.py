"""
╔══════════════════════════════════════════════════════════════════╗
║   PACKET_CAPTURE.PY - BẮT GÓI TIN THẬT TỪ NIC (SCAPY)        ║
╠══════════════════════════════════════════════════════════════════╣
║  Máy Ubuntu: 192.168.1.7                                        ║
║  Máy Kali  : 192.168.1.9 (attacker)                            ║
║  Interface : ens33 (bridge VMware)                              ║
╚══════════════════════════════════════════════════════════════════╝

CÁCH HOẠT ĐỘNG:
───────────────
  NIC (ens33)
      │  raw packets
      ▼
  AsyncSniffer (scapy)          ← chạy trong thread riêng
      │  mỗi packet → parse header
      ▼
  FlowAggregator                ← gom packets → flows theo 5-tuple
      │  khi flow hết (FIN/RST/timeout) → xuất 18 features
      ▼
  packet_queue                  ← Producer đẩy vào đây
      │
      ▼ (Consumer trong main_monitor.py lấy ra xử lý tiếp)

YÊU CẦU CHẠY:
──────────────
  sudo python main_monitor.py --mode live
  (scapy cần quyền root để sniff trên interface thật)

CÀI ĐẶT:
─────────
  pip install scapy
"""

import time
import queue
import threading
import logging
from collections import defaultdict

import numpy as np

# Scapy - cần cài: pip install scapy
try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP, ICMP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("[!] Scapy chưa được cài. Chạy: pip install scapy")

from feature_extractor import FlowAggregator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  AUTO-DETECT INTERFACE
# ═══════════════════════════════════════════════════════════════

def detect_interface():
    """
    Tự động tìm interface có địa chỉ IP thật (không phải loopback).
    Ubuntu + VMware bridge thường là ens33 hoặc eth0.

    Returns:
        str: Tên interface (vd: 'ens33', 'eth0')
    """
    try:
        import subprocess
        # Lấy interface đang có default route
        result = subprocess.run(
            ['ip', 'route', 'show', 'default'],
            capture_output=True, text=True
        )
        # Output dạng: "default via 192.168.1.1 dev ens33 proto ..."
        for part in result.stdout.split():
            if part not in ('default', 'via', 'dev', 'proto', 'src', 'metric'):
                if not part.startswith('192.') and not part.startswith('10.'):
                    prev = result.stdout.split()[result.stdout.split().index(part) - 1]
                    if prev == 'dev':
                        logger.info(f"[*] Auto-detected interface: {part}")
                        return part
    except Exception:
        pass

    # Fallback theo thứ tự phổ biến trên Ubuntu VMware
    for iface in ['ens33', 'eth0', 'enp0s3', 'ens32']:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            import fcntl, struct
            ifreq = struct.pack('16sH14s', iface.encode(), 2, b'\x00' * 14)
            import fcntl
            res = fcntl.ioctl(s.fileno(), 0x8915, ifreq)
            ip = socket.inet_ntoa(res[20:24])
            if ip and not ip.startswith('127.'):
                logger.info(f"[*] Found interface {iface} with IP {ip}")
                return iface
        except Exception:
            continue

    logger.warning("[!] Không tự detect được interface, dùng 'ens33'")
    return 'ens33'


# ═══════════════════════════════════════════════════════════════
#  PACKET PARSER
# ═══════════════════════════════════════════════════════════════

def parse_scapy_packet(pkt):
    """
    Parse một scapy packet thành dict chuẩn cho FlowAggregator.

    Input:  scapy Packet object
    Output: dict với các keys mà FlowAggregator.add_packet() cần,
            hoặc None nếu packet không có IP layer (ARP, v.v.)

    Các key bắt buộc:
        src_ip, dst_ip, src_port, dst_port, protocol,
        size, timestamp,
        syn_flag, ack_flag, fin_flag, rst_flag, psh_flag
    """
    # Chỉ xử lý IP packets (bỏ qua ARP, LLDP, ...)
    if not pkt.haslayer(IP):
        return None

    ip_layer = pkt[IP]

    packet_info = {
        'src_ip':    ip_layer.src,
        'dst_ip':    ip_layer.dst,
        'src_port':  0,
        'dst_port':  0,
        'protocol':  ip_layer.proto,   # 6=TCP, 17=UDP, 1=ICMP
        'size':      len(pkt),
        'timestamp': time.time(),
        # TCP flags mặc định = False
        'syn_flag':  False,
        'ack_flag':  False,
        'fin_flag':  False,
        'rst_flag':  False,
        'psh_flag':  False,
    }

    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        packet_info['src_port'] = tcp.sport
        packet_info['dst_port'] = tcp.dport
        # Parse TCP flags (dạng string: 'S', 'SA', 'FA', ...)
        flags = str(tcp.flags)
        packet_info['syn_flag'] = 'S' in flags
        packet_info['ack_flag'] = 'A' in flags
        packet_info['fin_flag'] = 'F' in flags
        packet_info['rst_flag'] = 'R' in flags
        packet_info['psh_flag'] = 'P' in flags

    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        packet_info['src_port'] = udp.sport
        packet_info['dst_port'] = udp.dport

    elif pkt.haslayer(ICMP):
        # ICMP không có port, dùng type/code thay thế
        icmp = pkt[ICMP]
        packet_info['dst_port'] = icmp.type   # ICMP type làm "port"

    return packet_info


# ═══════════════════════════════════════════════════════════════
#  LIVE PACKET CAPTURE CLASS
# ═══════════════════════════════════════════════════════════════

class LivePacketCapture:
    """
    ╔════════════════════════════════════════════════════════╗
    ║  LIVE PACKET CAPTURE                                  ║
    ╠════════════════════════════════════════════════════════╣
    ║  Bắt packet thật từ NIC, gom thành flows,            ║
    ║  đẩy feature vectors vào packet_queue.               ║
    ╚════════════════════════════════════════════════════════╝

    Sử dụng:
        capture = LivePacketCapture(packet_queue, iface='ens33')
        capture.start()
        ...
        capture.stop()
    """

    def __init__(self,
                 packet_queue: queue.Queue,
                 iface: str = None,
                 flow_timeout: float = 30.0,
                 stats=None):
        """
        Args:
            packet_queue: Queue để đẩy feature vectors (18-dim numpy array)
                          Consumer trong main_monitor.py sẽ lấy từ đây
            iface:        Tên interface (None = auto-detect)
            flow_timeout: Giây chờ trước khi export flow chưa kết thúc
            stats:        MonitorStats object (từ main_monitor.py) để ghi metrics
        """
        if not HAS_SCAPY:
            raise ImportError("Scapy chưa được cài. Chạy: pip install scapy")

        self.packet_queue = packet_queue
        self.iface = iface or detect_interface()
        self.flow_timeout = flow_timeout
        self.stats = stats

        self.flow_aggregator = FlowAggregator(flow_timeout=flow_timeout)
        self._sniffer = None
        self._timeout_thread = None
        self._stop_event = threading.Event()

        # Counters
        self.packets_seen = 0
        self.flows_exported = 0
        self.packets_dropped = 0   # Queue full

        logger.info(f"[LiveCapture] Interface: {self.iface} | "
                    f"Flow timeout: {flow_timeout}s")

    def _packet_callback(self, pkt):
        """
        Callback được scapy gọi mỗi khi có packet mới.
        Chạy trong thread của AsyncSniffer.

        LÝ THUYẾT: Tại sao không xử lý nặng ở đây?
        ──────────────────────────────────────────
        Callback phải trả về NHANH nhất có thể.
        Nếu xử lý chậm → scapy bỏ packet tiếp theo.
        Parse packet → FlowAggregator là đủ nhanh.
        AI inference thì để Consumer làm sau.
        """
        self.packets_seen += 1
        if self.stats:
            self.stats.record_capture()

        packet_info = parse_scapy_packet(pkt)
        if packet_info is None:
            return  # Không phải IP packet

        # Đưa vào FlowAggregator
        # Nếu flow kết thúc (FIN/RST) → trả về feature vector ngay
        features = self.flow_aggregator.add_packet(packet_info)
        if features is not None:
            self._push_features(features)

    def _push_features(self, features: np.ndarray):
        """
        Đẩy feature vector vào queue để Consumer xử lý.
        Dùng put_nowait để không block callback thread.
        """
        try:
            # Đẩy cùng capture_time để tính latency
            capture_time = time.time()
            self.packet_queue.put_nowait((features, capture_time))
            self.flows_exported += 1
            if self.stats:
                self.stats.record_processed()
        except queue.Full:
            self.packets_dropped += 1
            if self.packets_dropped % 100 == 1:
                logger.warning(f"[LiveCapture] Queue đầy! "
                                f"Dropped: {self.packets_dropped} flows")

    def _timeout_worker(self):
        """
        Thread riêng: định kỳ export các flows đã timeout.

        LÝ THUYẾT: Tại sao cần timeout?
        ────────────────────────────────
        UDP, ICMP không có FIN/RST → flow không bao giờ "kết thúc".
        Nhiều TCP connection cũng bị reset mà không gửi FIN.
        → Cần timeout để export các flow này sau X giây im lặng.
        """
        check_interval = min(self.flow_timeout / 2, 10.0)
        logger.info(f"[LiveCapture] Timeout worker: check mỗi {check_interval:.0f}s")

        while not self._stop_event.is_set():
            self._stop_event.wait(check_interval)
            if self._stop_event.is_set():
                break

            expired = self.flow_aggregator.check_timeouts()
            for features in expired:
                self._push_features(features)

            if expired:
                logger.debug(f"[LiveCapture] Exported {len(expired)} timed-out flows "
                             f"| Total flows: {self.flows_exported} "
                             f"| Packets seen: {self.packets_seen}")

    def start(self):
        """
        Bắt đầu capture.

        LÝ THUYẾT: AsyncSniffer
        ───────────────────────
        AsyncSniffer chạy scapy trong thread riêng biệt
        → không block main thread
        → main thread vẫn có thể check stop_event, cập nhật stats

        filter='ip': chỉ bắt IP packets, bỏ qua ARP, STP, ...
        store=False: không lưu packets vào RAM (tiết kiệm memory)
        prn=callback: gọi callback cho mỗi packet
        """
        logger.info(f"[LiveCapture] Bắt đầu sniff trên {self.iface}...")
        print(f"\n{'='*60}")
        print(f"  LIVE CAPTURE bắt đầu trên interface: {self.iface}")
        print(f"  Máy Ubuntu (victim): 192.168.1.7")
        print(f"  Máy Kali  (attacker): 192.168.1.9")
        print(f"  Chạy tấn công từ Kali để thấy alerts!")
        print(f"{'='*60}\n")

        # Khởi động timeout worker thread
        self._stop_event.clear()
        self._timeout_thread = threading.Thread(
            target=self._timeout_worker,
            daemon=True,
            name="FlowTimeoutWorker"
        )
        self._timeout_thread.start()

        # Khởi động AsyncSniffer
        self._sniffer = AsyncSniffer(
            iface=self.iface,
            filter='ip',            # Chỉ IP packets
            prn=self._packet_callback,
            store=False,            # Không lưu vào RAM
        )
        self._sniffer.start()
        logger.info(f"[LiveCapture] AsyncSniffer đang chạy...")

    def stop(self):
        """Dừng capture và export các flows còn lại."""
        logger.info("[LiveCapture] Đang dừng...")
        self._stop_event.set()

        if self._sniffer and self._sniffer.running:
            self._sniffer.stop()

        # Export nốt các flows còn trong bộ nhớ
        remaining = self.flow_aggregator.check_timeouts()
        for features in remaining:
            self._push_features(features)

        logger.info(f"[LiveCapture] Đã dừng. "
                    f"Packets: {self.packets_seen} | "
                    f"Flows exported: {self.flows_exported} | "
                    f"Dropped: {self.packets_dropped}")

    @property
    def is_running(self):
        return self._sniffer is not None and self._sniffer.running


# ═══════════════════════════════════════════════════════════════
#  HÀM PRODUCER CHO LIVE MODE (dùng trong main_monitor.py)
# ═══════════════════════════════════════════════════════════════

def live_producer_thread(packet_queue: queue.Queue,
                         stop_event: threading.Event,
                         stats=None,
                         iface: str = None,
                         flow_timeout: float = 30.0):
    """
    Producer thread cho live capture mode.
    Thay thế producer_thread() (simulated) trong main_monitor.py.

    Cách dùng trong main_monitor.py:
    ─────────────────────────────────
        from packet_capture import live_producer_thread

        producer = threading.Thread(
            target=live_producer_thread,
            args=(packet_queue, stop_event, stats),
            kwargs={'iface': 'ens33'},
            daemon=True
        )
        producer.start()

    Args:
        packet_queue: Queue shared với Consumer threads
        stop_event:   Event để signal dừng (từ main)
        stats:        MonitorStats object
        iface:        Interface name (None = auto-detect)
        flow_timeout: Giây timeout cho flow
    """
    capture = LivePacketCapture(
        packet_queue=packet_queue,
        iface=iface,
        flow_timeout=flow_timeout,
        stats=stats,
    )

    capture.start()

    # Chờ cho đến khi có signal dừng
    while not stop_event.is_set():
        stop_event.wait(timeout=1.0)

    capture.stop()
    logger.info("[LiveProducer] Đã thoát.")


# ═══════════════════════════════════════════════════════════════
#  TEST ĐƠN GIẢN (chạy trực tiếp để kiểm tra)
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    """
    Chạy: sudo python packet_capture.py
    Để kiểm tra capture có hoạt động không trước khi tích hợp.
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("╔" + "═"*58 + "╗")
    print("║  TEST LIVE PACKET CAPTURE                              ║")
    print("╚" + "═"*58 + "╝")

    if not HAS_SCAPY:
        print("[!] Scapy chưa cài. Chạy: pip install scapy")
        sys.exit(1)

    test_queue = queue.Queue(maxsize=1000)
    stop = threading.Event()

    capture = LivePacketCapture(
        packet_queue=test_queue,
        flow_timeout=10.0,   # 10 giây timeout cho test nhanh
    )

    capture.start()
    print(f"\n[*] Đang capture 30 giây...")
    print(f"[*] Từ Kali chạy: nmap -sS 192.168.1.7")
    print(f"[*] Nhấn Ctrl+C để dừng sớm\n")

    try:
        for i in range(30):
            time.sleep(1)
            q_size = test_queue.qsize()
            print(f"  [{i+1:02d}s] Packets seen: {capture.packets_seen:5d} | "
                  f"Flows exported: {capture.flows_exported:4d} | "
                  f"Queue: {q_size:3d}")
    except KeyboardInterrupt:
        print("\n[*] Ctrl+C — đang dừng...")

    capture.stop()

    # In một số features đã bắt được
    print(f"\n{'─'*60}")
    print(f"KẾT QUẢ: {test_queue.qsize()} flow feature vectors trong queue")
    count = 0
    while not test_queue.empty() and count < 5:
        features, ts = test_queue.get_nowait()
        print(f"  Flow {count+1}: shape={features.shape} | "
              f"dst_port={features[17]:.0f} | "
              f"syn={features[8]:.0f} | "
              f"bytes/s={features[3]:.1f}")
        count += 1
    print(f"{'─'*60}")
