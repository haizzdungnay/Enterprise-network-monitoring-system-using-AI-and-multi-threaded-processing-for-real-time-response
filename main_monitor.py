"""
╔══════════════════════════════════════════════════════════════════╗
║     MAIN_MONITOR.PY - HỆ THỐNG GIÁM SÁT MẠNG CHÍNH           ║
╠══════════════════════════════════════════════════════════════════╣
║  Implement kiến trúc Producer-Consumer với AI Inference         ║
║  Hỗ trợ: Single-thread, Multi-thread, Multi-process           ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
  LÝ THUYẾT TỔNG QUAN: Producer-Consumer Pattern
═══════════════════════════════════════════════════════════════════

PRODUCER-CONSUMER là một design pattern kinh điển trong concurrent
programming, đặc biệt phù hợp cho data pipeline:

  Producer ──▶ [Queue/Buffer] ──▶ Consumer

TẠI SAO CẦN PATTERN NÀY?
─────────────────────────
1. DECOUPLING: Producer không cần biết Consumer xử lý gì
   → Dễ thay đổi, test từng phần riêng

2. BUFFERING: Queue hấp thụ sự chênh lệch tốc độ
   Ví dụ: Network burst 10,000 packets/sec nhưng AI chỉ xử lý
   2,000 packets/sec → Queue buffer tạm giữ phần dư

3. LOAD BALANCING: Nhiều Consumers cùng lấy từ 1 Queue
   → Tự động cân bằng tải (Consumer nào xong trước lấy tiếp)

4. BACKPRESSURE: Khi Queue đầy, Producer bị block
   → Tránh system overload (OOM, crash)

KIẾN TRÚC CỤ THỂ TRONG PROJECT:
────────────────────────────────
  ┌──────────┐     ┌──────────────┐     ┌───────────┐
  │ Producer │────▶│ Packet Queue │────▶│ Consumer  │
  │ (Capture)│     │ (thread-safe)│     │ (Extract) │
  └──────────┘     └──────────────┘     └─────┬─────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  Feature Queue     │
                                    │  (batch buffer)    │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  AI Worker Pool    │
                                    │  (parallel infer.) │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  Results/Alerts    │
                                    └───────────────────┘

═══════════════════════════════════════════════════════════════════
  LÝ THUYẾT: THREAD SAFETY
═══════════════════════════════════════════════════════════════════

Khi nhiều threads truy cập cùng 1 data structure:
  - Race condition: 2 threads đọc/ghi cùng lúc → data corrupt
  - Solution: Queue (đã thread-safe), Lock, Semaphore

Python Queue module:
  - queue.Queue: Thread-safe (dùng internal Lock)
  - queue.put(): Thêm item (block nếu queue đầy)
  - queue.get(): Lấy item (block nếu queue rỗng)
  - queue.put(item, timeout=1): Thêm với timeout
  - queue.get(timeout=1): Lấy với timeout

Multiprocessing Queue:
  - multiprocessing.Queue: Process-safe (dùng Pipe + Lock)
  - Chậm hơn thread Queue (serialize/deserialize data qua Pipe)
  - Nhưng không bị GIL → true parallelism
"""

import os
import sys
import time
import queue
import threading
import multiprocessing
import logging
import json
import signal
import numpy as np
import joblib

import config
from feature_extractor import (
    PacketFeatureExtractor,
    generate_simulated_packet,
    generate_simulated_features,
)
from ai_model import IDSModel

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, mode='w')
    ]
)
logger = logging.getLogger(__name__)


class MonitorStats:
    """
    ╔════════════════════════════════════════════════════════╗
    ║  Thread-safe statistics collector                     ║
    ╠════════════════════════════════════════════════════════╣
    ║  Thu thập metrics: throughput, latency, counts        ║
    ║  Sử dụng Lock để đảm bảo thread-safety               ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Lock (Mutex)
    ───────────────────────
    Lock đảm bảo chỉ 1 thread truy cập critical section tại 1 thời điểm.

    with self.lock:           # Acquire lock (block nếu thread khác đang giữ)
        self.counter += 1     # Critical section - chỉ 1 thread thực hiện
                              # Release lock (tự động khi exit `with`)

    Nếu KHÔNG dùng Lock:
      Thread A: read counter=5, compute 5+1=6, write counter=6
      Thread B: read counter=5 (ĐỒNG THỜI!), compute 5+1=6, write counter=6
      → Counter = 6 thay vì 7 (Lost Update!)
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.packets_captured = 0
        self.packets_processed = 0
        self.ai_predictions = 0
        self.attacks_detected = 0
        self.total_latency = 0.0
        self.latency_samples = 0
        self.start_time = time.time()
        self.latencies = []

    def record_capture(self):
        with self.lock:
            self.packets_captured += 1

    def record_processed(self):
        with self.lock:
            self.packets_processed += 1

    def record_prediction(self, is_attack, latency):
        with self.lock:
            self.ai_predictions += 1
            if is_attack:
                self.attacks_detected += 1
            self.total_latency += latency
            self.latency_samples += 1
            self.latencies.append(latency)

    def get_summary(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            avg_latency = (self.total_latency / self.latency_samples
                          if self.latency_samples > 0 else 0)

            # Percentile latencies
            p50 = np.percentile(self.latencies, 50) if self.latencies else 0
            p95 = np.percentile(self.latencies, 95) if self.latencies else 0
            p99 = np.percentile(self.latencies, 99) if self.latencies else 0

            return {
                'elapsed_sec': elapsed,
                'packets_captured': self.packets_captured,
                'packets_processed': self.packets_processed,
                'ai_predictions': self.ai_predictions,
                'attacks_detected': self.attacks_detected,
                'throughput_capture': self.packets_captured / max(elapsed, 0.001),
                'throughput_inference': self.ai_predictions / max(elapsed, 0.001),
                'avg_latency_ms': avg_latency * 1000,
                'p50_latency_ms': p50 * 1000,
                'p95_latency_ms': p95 * 1000,
                'p99_latency_ms': p99 * 1000,
            }


# ═══════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░ MODE 1: SINGLE-THREAD (BASELINE) ░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════

def run_single_thread(model, scaler, num_packets, stats):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  SINGLE-THREAD MODE                                    ║
    ╠════════════════════════════════════════════════════════╣
    ║  Tuần tự: Capture → Extract → Predict → Repeat       ║
    ║  Dùng làm BASELINE để so sánh                        ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Tại sao single-thread là baseline?
    ──────────────────────────────────────────────
    - Đơn giản nhất, dễ debug
    - Không có overhead từ threading/multiprocessing
    - Nhưng: CPU chỉ dùng 1 core, không tận dụng hardware
    - Khi capture → CPU idle (chờ I/O)
    - Khi predict → NIC buffer có thể overflow (mất packet)
    """
    logger.info("Starting SINGLE-THREAD mode...")
    extractor = PacketFeatureExtractor()
    batch_buffer = []

    for i in range(num_packets):
        # 1. Capture (simulated)
        capture_time = time.time()
        packet = generate_simulated_packet()
        stats.record_capture()

        # 2. Extract features
        features = generate_simulated_features()
        stats.record_processed()

        # 3. Accumulate batch
        batch_buffer.append((features, capture_time))

        # 4. Predict when batch is full
        if len(batch_buffer) >= config.BATCH_SIZE:
            X_batch = np.array([f[0] for f in batch_buffer])
            timestamps = [f[1] for f in batch_buffer]

            # Scale features
            X_scaled = scaler.transform(X_batch)

            # Predict
            labels, probas = model.batch_predict(X_scaled)

            # Record results
            now = time.time()
            for j, (label, ts) in enumerate(zip(labels, timestamps)):
                latency = now - ts
                stats.record_prediction(
                    is_attack=(label == 1),
                    latency=latency
                )

            batch_buffer = []

    # Process remaining
    if batch_buffer:
        X_batch = np.array([f[0] for f in batch_buffer])
        timestamps = [f[1] for f in batch_buffer]
        X_scaled = scaler.transform(X_batch)
        labels, probas = model.batch_predict(X_scaled)
        now = time.time()
        for j, (label, ts) in enumerate(zip(labels, timestamps)):
            stats.record_prediction(is_attack=(label == 1), latency=now - ts)

    logger.info("Single-thread mode completed.")


# ═══════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░ MODE 2: MULTI-THREAD ░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════

def producer_thread(packet_queue, num_packets, stats, stop_event):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  PRODUCER THREAD - Capture packets                    ║
    ╠════════════════════════════════════════════════════════╣
    ║  Vai trò: Bắt packet từ NIC (hoặc simulate)          ║
    ║  Đẩy vào packet_queue để Consumer xử lý              ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Producer Design
    ──────────────────────────
    Producer nên:
    1. Capture càng nhanh càng tốt (không làm thêm gì)
    2. Dùng queue.put(timeout=X) để không block vĩnh viễn
    3. Gửi SENTINEL value (None) khi hoàn thành → Consumer biết dừng
    4. Handle backpressure: nếu queue đầy → drop packet hoặc block

    BACKPRESSURE STRATEGIES:
    - Block (default): Producer chờ queue có chỗ → không mất packet
    - Drop: Producer bỏ qua packet → mất data nhưng không chậm
    - Sampling: Chỉ đưa 1/N packets vào queue → compromise
    """
    logger.info(f"Producer started. Target: {num_packets} packets")

    for i in range(num_packets):
        if stop_event.is_set():
            break

        packet = generate_simulated_packet()
        capture_time = time.time()
        stats.record_capture()

        try:
            # put với timeout để không block vĩnh viễn
            packet_queue.put((packet, capture_time), timeout=2.0)
        except queue.Full:
            logger.warning(f"Packet queue FULL! Dropping packet #{i}")
            # Trong production: counter dropped_packets += 1

    # Sentinel: báo cho consumers biết producer đã xong
    packet_queue.put(None)
    logger.info(f"Producer finished. Captured: {stats.packets_captured}")


def consumer_thread(packet_queue, feature_queue, stats, stop_event):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  CONSUMER THREAD - Feature Extraction                 ║
    ╠════════════════════════════════════════════════════════╣
    ║  Vai trò: Lấy packet từ queue, trích xuất features   ║
    ║  Nhóm features thành batch, đẩy vào feature_queue    ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Consumer Design
    ──────────────────────────
    Consumer nên:
    1. Lấy packet từ queue (blocking get với timeout)
    2. Xử lý (extract features) - NHANH NHẤT CÓ THỂ
    3. Batch features lại trước khi đẩy cho AI worker
    4. Handle sentinel (None) → gửi remaining batch + sentinel tiếp

    BATCH ACCUMULATION:
    - Gom features cho đến khi đủ batch_size HOẶC timeout
    - Timeout cần thiết: nếu traffic chậm, không chờ mãi
    - Trade-off: batch lớn = throughput cao, latency cao
    """
    logger.info("Consumer started.")
    batch_buffer = []
    last_batch_time = time.time()

    while not stop_event.is_set():
        try:
            item = packet_queue.get(timeout=0.5)
        except queue.Empty:
            # Check batch timeout
            if batch_buffer and (time.time() - last_batch_time) > config.BATCH_TIMEOUT:
                feature_queue.put(batch_buffer)
                batch_buffer = []
                last_batch_time = time.time()
            continue

        # Sentinel check
        if item is None:
            if batch_buffer:
                feature_queue.put(batch_buffer)
            feature_queue.put(None)  # Forward sentinel
            logger.info("Consumer received sentinel. Shutting down.")
            break

        packet, capture_time = item

        # Extract features
        features = generate_simulated_features()
        stats.record_processed()

        batch_buffer.append((features, capture_time))

        # Flush batch khi đủ size
        if len(batch_buffer) >= config.BATCH_SIZE:
            feature_queue.put(batch_buffer)
            batch_buffer = []
            last_batch_time = time.time()


def ai_worker_thread(worker_id, feature_queue, model, scaler, stats, stop_event):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  AI WORKER THREAD - Inference                         ║
    ╠════════════════════════════════════════════════════════╣
    ║  Vai trò: Lấy batch features, chạy AI prediction     ║
    ║  Có thể chạy nhiều workers song song (Worker Pool)   ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Worker Pool Pattern
    ──────────────────────────────
    Thay vì 1 worker xử lý tất cả, dùng N workers:

      Feature Queue ──▶ Worker 1
                   ──▶ Worker 2
                   ──▶ Worker 3
                   ──▶ Worker N

    Mỗi worker lấy 1 batch, predict, rồi lấy batch tiếp theo.
    Queue tự động load balance: worker nào xong trước lấy tiếp.

    SỐ WORKERS TỐI ƯU:
    - CPU-bound (inference): N = số CPU cores
    - I/O-bound (nếu model trên GPU): N = 1-2 (GPU tự parallel)
    - Quá nhiều workers: Context switching overhead > performance gain

    LƯU Ý VỀ GIL:
    Python GIL hạn chế true parallelism cho CPU-bound tasks.
    sklearn/numpy TỰ GIẢI PHÓNG GIL khi chạy C/Fortran code bên dưới.
    → AI inference CÓ THỂ parallel thực sự dù dùng threading!
    """
    logger.info(f"AI Worker {worker_id} started.")

    while not stop_event.is_set():
        try:
            batch = feature_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Sentinel check
        if batch is None:
            feature_queue.put(None)  # Re-queue for other workers
            logger.info(f"AI Worker {worker_id} shutting down.")
            break

        # Prepare batch
        X_batch = np.array([item[0] for item in batch])
        timestamps = [item[1] for item in batch]

        # Scale & Predict
        try:
            X_scaled = scaler.transform(X_batch)
            labels, probas = model.batch_predict(X_scaled)

            # Record results
            now = time.time()
            for j, (label, ts) in enumerate(zip(labels, timestamps)):
                latency = now - ts
                is_attack = (label == 1)
                stats.record_prediction(is_attack=is_attack, latency=latency)

                # Alert on high-confidence attacks
                if is_attack and probas[j][1] > config.ALERT_THRESHOLD:
                    logger.warning(
                        f"🚨 ATTACK DETECTED! Confidence: {probas[j][1]:.2%}"
                    )
        except Exception as e:
            logger.error(f"AI Worker {worker_id} error: {e}")


def run_multi_thread(model, scaler, num_packets, stats):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  MULTI-THREAD ORCHESTRATOR                            ║
    ╠════════════════════════════════════════════════════════╣
    ║  Khởi tạo và quản lý tất cả threads                  ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Thread Management
    ────────────────────────────
    1. Tạo shared queues (thread-safe)
    2. Start producer thread(s)
    3. Start consumer thread(s)
    4. Start AI worker thread(s)
    5. Wait for all threads to finish (join)

    GRACEFUL SHUTDOWN:
    - stop_event: threading.Event() dùng để signal tất cả threads dừng
    - Sentinel (None): gửi qua queue để báo "không còn data"
    - join(): Main thread chờ tất cả threads kết thúc
    """
    logger.info("Starting MULTI-THREAD mode...")
    logger.info(f"  Consumers: {config.NUM_CONSUMER_THREADS}, "
                f"AI Workers: {config.NUM_AI_WORKERS}")

    # Shared queues
    packet_queue = queue.Queue(maxsize=config.PACKET_QUEUE_SIZE)
    feature_queue = queue.Queue(maxsize=config.FEATURE_QUEUE_SIZE)
    stop_event = threading.Event()

    threads = []

    # Start Producer
    t = threading.Thread(
        target=producer_thread,
        args=(packet_queue, num_packets, stats, stop_event),
        name="Producer-0",
        daemon=True
    )
    threads.append(t)

    # Start Consumers
    for i in range(config.NUM_CONSUMER_THREADS):
        t = threading.Thread(
            target=consumer_thread,
            args=(packet_queue, feature_queue, stats, stop_event),
            name=f"Consumer-{i}",
            daemon=True
        )
        threads.append(t)

    # Start AI Workers
    for i in range(config.NUM_AI_WORKERS):
        t = threading.Thread(
            target=ai_worker_thread,
            args=(i, feature_queue, model, scaler, stats, stop_event),
            name=f"AIWorker-{i}",
            daemon=True
        )
        threads.append(t)

    # Start all
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join(timeout=120)

    logger.info("Multi-thread mode completed.")


# ═══════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░ MODE 3: MULTI-PROCESS ░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════

def producer_process_fn(packet_queue, num_packets, captured_count, stop_event):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  PRODUCER PROCESS                                      ║
    ╠════════════════════════════════════════════════════════╣
    ║  Giống producer_thread nhưng chạy trong process riêng  ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Process vs Thread cho Producer
    ─────────────────────────────────────────
    - Thread: Share memory → queue operations nhanh
    - Process: Isolated memory → cần serialize data qua Pipe
    - Producer chủ yếu I/O-bound → Thread thường đủ
    - Nhưng Process không bị GIL → capture thread không bị
      interrupt bởi Python GIL khi consumer đang xử lý

    SHARED STATE giữa processes:
    - multiprocessing.Value: shared counter (atomic operations)
    - multiprocessing.Queue: process-safe queue
    - Manager: shared dict, list (nhưng chậm)
    """
    for i in range(num_packets):
        if stop_event.is_set():
            break
        packet = generate_simulated_packet()
        capture_time = time.time()
        with captured_count.get_lock():
            captured_count.value += 1
        try:
            packet_queue.put((packet, capture_time), timeout=2.0)
        except:
            pass

    packet_queue.put(None)


def consumer_process_fn(packet_queue, feature_queue, processed_count, stop_event):
    """Consumer process: extract features from packets."""
    batch_buffer = []
    last_batch_time = time.time()

    while not stop_event.is_set():
        try:
            item = packet_queue.get(timeout=0.5)
        except:
            if batch_buffer and (time.time() - last_batch_time) > config.BATCH_TIMEOUT:
                feature_queue.put(batch_buffer)
                batch_buffer = []
                last_batch_time = time.time()
            continue

        if item is None:
            if batch_buffer:
                feature_queue.put(batch_buffer)
            feature_queue.put(None)
            break

        packet, capture_time = item
        features = generate_simulated_features()
        with processed_count.get_lock():
            processed_count.value += 1

        batch_buffer.append((features, capture_time))
        if len(batch_buffer) >= config.BATCH_SIZE:
            feature_queue.put(batch_buffer)
            batch_buffer = []
            last_batch_time = time.time()


def ai_worker_process_fn(worker_id, feature_queue, model_path, scaler_path,
                          prediction_count, attack_count, latency_sum,
                          latency_count, stop_event):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  AI WORKER PROCESS                                     ║
    ╠════════════════════════════════════════════════════════╣
    ║  Mỗi process load model riêng (isolated memory)        ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Model Loading trong Multiprocessing
    ──────────────────────────────────────────────
    Vì mỗi process có memory space riêng:
    - KHÔNG THỂ share model object giữa processes
    - Mỗi worker phải load model từ file riêng
    - Trade-off: Tốn thêm RAM (N copies of model)
    - Nhưng: True parallelism, không bị GIL

    So sánh memory usage:
    - Threading: 1 model × 1 copy = ~50MB RAM
    - Multiprocessing (4 workers): 1 model × 4 copies = ~200MB RAM
    """
    model = IDSModel()
    model.load(model_path)
    scaler = joblib.load(scaler_path)

    while not stop_event.is_set():
        try:
            batch = feature_queue.get(timeout=0.5)
        except:
            continue

        if batch is None:
            feature_queue.put(None)
            break

        X_batch = np.array([item[0] for item in batch])
        timestamps = [item[1] for item in batch]

        try:
            X_scaled = scaler.transform(X_batch)
            labels, probas = model.batch_predict(X_scaled)

            now = time.time()
            for j, (label, ts) in enumerate(zip(labels, timestamps)):
                latency = now - ts
                with prediction_count.get_lock():
                    prediction_count.value += 1
                if label == 1:
                    with attack_count.get_lock():
                        attack_count.value += 1
                with latency_sum.get_lock():
                    latency_sum.value += latency
                with latency_count.get_lock():
                    latency_count.value += 1
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")


def run_multi_process(model_path, scaler_path, num_packets):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  MULTI-PROCESS ORCHESTRATOR                            ║
    ╠════════════════════════════════════════════════════════╣
    ║  Quản lý tất cả processes                             ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Multiprocessing Challenges
    ────────────────────────────────────
    1. IPC (Inter-Process Communication): Chậm hơn thread queue
       - Data phải serialize (pickle) → gửi qua Pipe → deserialize
       - Overhead ~10-100μs per message (vs ~1μs cho thread queue)

    2. Shared State: Cần dùng multiprocessing.Value (atomic)
       - Mỗi Value có lock riêng
       - Chậm hơn thread shared variables

    3. Process startup: Tạo process mất ~50-100ms
       (vs ~1ms cho thread)

    KHI NÀO DÙNG MULTIPROCESSING?
    - CPU-bound tasks chiếm chủ đạo (AI inference nặng)
    - Cần true parallelism (không bị GIL)
    - Acceptable overhead cho IPC
    - Đủ RAM cho multiple model copies
    """
    logger.info("Starting MULTI-PROCESS mode...")

    # Shared counters (process-safe)
    captured_count = multiprocessing.Value('i', 0)
    processed_count = multiprocessing.Value('i', 0)
    prediction_count = multiprocessing.Value('i', 0)
    attack_count = multiprocessing.Value('i', 0)
    latency_sum = multiprocessing.Value('d', 0.0)
    latency_count = multiprocessing.Value('i', 0)
    stop_event = multiprocessing.Event()

    # Process-safe queues
    packet_queue = multiprocessing.Queue(maxsize=config.PACKET_QUEUE_SIZE)
    feature_queue = multiprocessing.Queue(maxsize=config.FEATURE_QUEUE_SIZE)

    processes = []
    start_time = time.time()

    # Producer process
    p = multiprocessing.Process(
        target=producer_process_fn,
        args=(packet_queue, num_packets, captured_count, stop_event),
        name="Producer-0"
    )
    processes.append(p)

    # Consumer processes
    for i in range(config.NUM_CONSUMER_THREADS):
        p = multiprocessing.Process(
            target=consumer_process_fn,
            args=(packet_queue, feature_queue, processed_count, stop_event),
            name=f"Consumer-{i}"
        )
        processes.append(p)

    # AI Worker processes
    num_ai = max(1, config.NUM_AI_WORKERS)
    for i in range(num_ai):
        p = multiprocessing.Process(
            target=ai_worker_process_fn,
            args=(i, feature_queue, model_path, scaler_path,
                  prediction_count, attack_count, latency_sum,
                  latency_count, stop_event),
            name=f"AIWorker-{i}"
        )
        processes.append(p)

    # Start all
    for p in processes:
        p.start()

    # Wait
    for p in processes:
        p.join(timeout=120)

    elapsed = time.time() - start_time
    avg_latency = (latency_sum.value / max(latency_count.value, 1))

    return {
        'elapsed_sec': elapsed,
        'packets_captured': captured_count.value,
        'packets_processed': processed_count.value,
        'ai_predictions': prediction_count.value,
        'attacks_detected': attack_count.value,
        'throughput_capture': captured_count.value / max(elapsed, 0.001),
        'throughput_inference': prediction_count.value / max(elapsed, 0.001),
        'avg_latency_ms': avg_latency * 1000,
    }


# ═══════════════════════════════════════════════════════════════
# STATS PRINTER
# ═══════════════════════════════════════════════════════════════

def print_stats(stats_dict, mode_name):
    """In ra kết quả đẹp."""
    print(f"\n{'═' * 60}")
    print(f"  KẾT QUẢ: {mode_name}")
    print(f"{'═' * 60}")
    print(f"  Thời gian chạy:      {stats_dict['elapsed_sec']:.2f}s")
    print(f"  Packets captured:    {stats_dict['packets_captured']:,}")
    print(f"  Packets processed:   {stats_dict['packets_processed']:,}")
    print(f"  AI predictions:      {stats_dict['ai_predictions']:,}")
    print(f"  Attacks detected:    {stats_dict['attacks_detected']:,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Capture throughput:  {stats_dict['throughput_capture']:,.0f} pkts/sec")
    print(f"  Inference throughput:{stats_dict['throughput_inference']:,.0f} preds/sec")
    print(f"  Avg latency:         {stats_dict['avg_latency_ms']:.2f} ms")
    if 'p95_latency_ms' in stats_dict:
        print(f"  P50 latency:         {stats_dict['p50_latency_ms']:.2f} ms")
        print(f"  P95 latency:         {stats_dict['p95_latency_ms']:.2f} ms")
        print(f"  P99 latency:         {stats_dict['p99_latency_ms']:.2f} ms")
    print(f"{'═' * 60}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 58 + "╗")
    print("║    HỆ THỐNG GIÁM SÁT MẠNG - ENTERPRISE NETWORK IDS    ║")
    print("╚" + "═" * 58 + "╝")
    config.print_config()

    # Check model exists
    if not os.path.exists(config.MODEL_PATH):
        print("\n[!] Model chưa được train. Chạy: python train_model.py")
        print("[*] Đang tự động train model...")
        os.system(f"{sys.executable} train_model.py")

    # Load model
    print("\n[*] Loading model...")
    model = IDSModel()
    model.load()
    scaler = joblib.load(config.SCALER_PATH)

    num_packets = config.BENCHMARK_NUM_PACKETS

    # ── Run all 3 modes ──
    all_results = {}

    # Mode 1: Single Thread
    print("\n" + "▓" * 60)
    print("  RUNNING: Single-Thread Mode")
    print("▓" * 60)
    stats_st = MonitorStats()
    run_single_thread(model, scaler, num_packets, stats_st)
    result_st = stats_st.get_summary()
    print_stats(result_st, "SINGLE-THREAD")
    all_results['single_thread'] = result_st

    # Mode 2: Multi-Thread
    print("\n" + "▓" * 60)
    print("  RUNNING: Multi-Thread Mode")
    print("▓" * 60)
    stats_mt = MonitorStats()
    run_multi_thread(model, scaler, num_packets, stats_mt)
    result_mt = stats_mt.get_summary()
    print_stats(result_mt, "MULTI-THREAD")
    all_results['multi_thread'] = result_mt

    # Mode 3: Multi-Process
    print("\n" + "▓" * 60)
    print("  RUNNING: Multi-Process Mode")
    print("▓" * 60)
    result_mp = run_multi_process(config.MODEL_PATH, config.SCALER_PATH, num_packets)
    print_stats(result_mp, "MULTI-PROCESS")
    all_results['multi_process'] = result_mp

    # Save results
    results_path = os.path.join(config.RESULTS_DIR, "monitor_results.json")
    serializable = {}
    for mode, data in all_results.items():
        serializable[mode] = {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                               for k, v in data.items()}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n[+] Results saved: {results_path}")

    return all_results


if __name__ == "__main__":
    main()
