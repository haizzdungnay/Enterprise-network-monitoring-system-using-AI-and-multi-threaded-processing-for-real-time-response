"""
╔══════════════════════════════════════════════════════════════════╗
║     BENCHMARK.PY - SO SÁNH PERFORMANCE CÁC CHẾ ĐỘ             ║
╠══════════════════════════════════════════════════════════════════╣
║  Đo lường: Throughput, Latency, Scalability                    ║
║  So sánh: Single-thread, Multi-thread, Multi-process           ║
║  Biến thiên: Queue size, Batch size, Worker count              ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
LÝ THUYẾT: Benchmarking Methodology
═══════════════════════════════════════════════════════════════════

NGUYÊN TẮC ĐO PERFORMANCE:
──────────────────────────
1. WARMUP: Bỏ N samples đầu tiên (cache cold, JIT compilation)
2. REPEAT: Đo nhiều lần, lấy trung bình (giảm variance)
3. ISOLATE: Đo từng component riêng trước khi đo end-to-end
4. CONTROL: Cùng workload, cùng hardware cho mọi configurations

METRICS QUAN TRỌNG:
───────────────────
1. THROUGHPUT (packets/sec hoặc predictions/sec):
   - Số lượng work units hoàn thành mỗi giây
   - Metric chính cho batch processing
   - Higher = better

2. LATENCY (milliseconds):
   - Thời gian từ khi packet arrive đến khi có prediction
   - End-to-end: capture → extract → queue → batch → predict → alert
   - Percentiles: P50 (median), P95, P99
   - P99 quan trọng cho SLA: "99% requests dưới Xms"
   - Lower = better

3. SCALABILITY:
   - Throughput thay đổi thế nào khi tăng resources?
   - Linear scaling (2x workers → 2x throughput) là lý tưởng
   - Sublinear (2x workers → 1.5x throughput) do overhead
   - Negative scaling: quá nhiều workers → chậm hơn!

AMDAHL'S LAW:
────────────
Speedup tối đa khi parallelize:
  S = 1 / ((1-P) + P/N)
  Trong đó:
    P = phần có thể parallel (0-1)
    N = số processors
    S = speedup

Ví dụ: P=0.9 (90% parallelizable), N=4 cores:
  S = 1 / (0.1 + 0.9/4) = 1 / 0.325 = 3.08x
  → Dù có 4 cores, chỉ nhanh hơn 3.08x (không phải 4x)
  → 10% sequential code giới hạn toàn bộ system!
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import queue
import threading
import multiprocessing
import numpy as np
import joblib

import config
from feature_extractor import generate_simulated_features
from ai_model import IDSModel


def benchmark_inference_only(model, scaler, num_samples, batch_size):
    """Benchmark chỉ riêng inference speed (không queue, không threading)."""
    # Generate data
    X = np.array([generate_simulated_features() for _ in range(num_samples)])
    X_scaled = scaler.transform(X)

    # Warmup
    _ = model.predict(X_scaled[:min(100, len(X_scaled))])

    # Benchmark
    start = time.time()
    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i+batch_size]
        model.predict(batch)
    elapsed = time.time() - start

    return {
        'throughput': num_samples / elapsed,
        'total_time': elapsed,
        'avg_latency_ms': (elapsed / num_samples) * 1000,
    }


def benchmark_pipeline_single_thread(model, scaler, num_samples, batch_size):
    """Full pipeline single-thread."""
    latencies = []
    batch_buffer = []
    predictions = 0

    start = time.time()
    for i in range(num_samples):
        capture_time = time.time()
        features = generate_simulated_features()
        batch_buffer.append((features, capture_time))

        if len(batch_buffer) >= batch_size:
            X_batch = np.array([f[0] for f in batch_buffer])
            timestamps = [f[1] for f in batch_buffer]
            X_scaled = scaler.transform(X_batch)
            labels = model.predict(X_scaled)

            now = time.time()
            for ts in timestamps:
                latencies.append(now - ts)
            predictions += len(labels)
            batch_buffer = []

    # Remaining
    if batch_buffer:
        X_batch = np.array([f[0] for f in batch_buffer])
        timestamps = [f[1] for f in batch_buffer]
        X_scaled = scaler.transform(X_batch)
        labels = model.predict(X_scaled)
        now = time.time()
        for ts in timestamps:
            latencies.append(now - ts)
        predictions += len(labels)

    elapsed = time.time() - start

    return {
        'throughput': predictions / elapsed,
        'total_time': elapsed,
        'avg_latency_ms': np.mean(latencies) * 1000,
        'p50_latency_ms': np.percentile(latencies, 50) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        'predictions': predictions,
    }


def benchmark_pipeline_multi_thread(model, scaler, num_samples, batch_size,
                                     num_consumers, num_ai_workers):
    """Full pipeline multi-thread."""
    packet_queue = queue.Queue(maxsize=config.PACKET_QUEUE_SIZE)
    feature_queue = queue.Queue(maxsize=config.FEATURE_QUEUE_SIZE)
    stop_event = threading.Event()

    latencies = []
    lat_lock = threading.Lock()
    prediction_count = [0]

    def _producer():
        for i in range(num_samples):
            if stop_event.is_set():
                break
            features = generate_simulated_features()
            capture_time = time.time()
            try:
                packet_queue.put((features, capture_time), timeout=2)
            except queue.Full:
                pass
        packet_queue.put(None)

    def _consumer():
        batch_buf = []
        while not stop_event.is_set():
            try:
                item = packet_queue.get(timeout=0.5)
            except queue.Empty:
                if batch_buf:
                    feature_queue.put(batch_buf)
                    batch_buf = []
                continue

            if item is None:
                if batch_buf:
                    feature_queue.put(batch_buf)
                feature_queue.put(None)
                break

            batch_buf.append(item)
            if len(batch_buf) >= batch_size:
                feature_queue.put(batch_buf)
                batch_buf = []

    def _ai_worker():
        while not stop_event.is_set():
            try:
                batch = feature_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if batch is None:
                feature_queue.put(None)
                break

            X = np.array([item[0] for item in batch])
            timestamps = [item[1] for item in batch]
            try:
                X_scaled = scaler.transform(X)
                labels = model.predict(X_scaled)
                now = time.time()
                with lat_lock:
                    for ts in timestamps:
                        latencies.append(now - ts)
                    prediction_count[0] += len(labels)
            except:
                pass

    threads = []
    start = time.time()

    # Producer
    t = threading.Thread(target=_producer, daemon=True)
    threads.append(t)

    # Consumers
    for _ in range(num_consumers):
        t = threading.Thread(target=_consumer, daemon=True)
        threads.append(t)

    # AI Workers
    for _ in range(num_ai_workers):
        t = threading.Thread(target=_ai_worker, daemon=True)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    elapsed = time.time() - start
    preds = prediction_count[0]

    result = {
        'throughput': preds / max(elapsed, 0.001),
        'total_time': elapsed,
        'predictions': preds,
    }

    if latencies:
        result.update({
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        })
    else:
        result.update({
            'avg_latency_ms': 0, 'p50_latency_ms': 0,
            'p95_latency_ms': 0, 'p99_latency_ms': 0,
        })

    return result


def mp_ai_worker_fn(feature_queue, model_path, scaler_path,
                     result_queue, stop_event):
    """AI worker for multiprocessing benchmark."""
    model = IDSModel()
    model.load(model_path)
    scaler = joblib.load(scaler_path)
    local_latencies = []
    local_preds = 0

    while not stop_event.is_set():
        try:
            batch = feature_queue.get(timeout=0.5)
        except:
            continue
        if batch is None:
            feature_queue.put(None)
            break

        X = np.array([item[0] for item in batch])
        timestamps = [item[1] for item in batch]
        try:
            X_scaled = scaler.transform(X)
            labels = model.predict(X_scaled)
            now = time.time()
            for ts in timestamps:
                local_latencies.append(now - ts)
            local_preds += len(labels)
        except:
            pass

    result_queue.put((local_preds, local_latencies))


def mp_producer_fn(packet_queue, num_samples, stop_event, num_consumers):
    """Producer process function (must be top-level for Windows spawn)."""
    for _ in range(num_samples):
        if stop_event.is_set():
            break
        features = generate_simulated_features()
        capture_time = time.time()
        try:
            packet_queue.put((features, capture_time), timeout=2)
        except:
            pass

    # Send one sentinel per consumer so all consumers can terminate cleanly.
    for _ in range(num_consumers):
        packet_queue.put(None)


def mp_consumer_fn(packet_queue, feature_queue, batch_size, stop_event, num_ai_workers):
    """Consumer process function (must be top-level for Windows spawn)."""
    batch_buf = []
    while not stop_event.is_set():
        try:
            item = packet_queue.get(timeout=0.5)
        except:
            if batch_buf:
                feature_queue.put(batch_buf)
                batch_buf = []
            continue

        if item is None:
            if batch_buf:
                feature_queue.put(batch_buf)

            # Forward stop signal to AI workers.
            for _ in range(num_ai_workers):
                feature_queue.put(None)
            break

        batch_buf.append(item)
        if len(batch_buf) >= batch_size:
            feature_queue.put(batch_buf)
            batch_buf = []


def benchmark_pipeline_multi_process(model_path, scaler_path, num_samples,
                                      batch_size, num_consumers, num_ai_workers):
    """Full pipeline multi-process."""
    packet_queue = multiprocessing.Queue(maxsize=config.PACKET_QUEUE_SIZE)
    feature_queue = multiprocessing.Queue(maxsize=config.FEATURE_QUEUE_SIZE)
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    processes = []
    start = time.time()

    p = multiprocessing.Process(
        target=mp_producer_fn,
        args=(packet_queue, num_samples, stop_event, num_consumers)
    )
    processes.append(p)

    for _ in range(num_consumers):
        p = multiprocessing.Process(
            target=mp_consumer_fn,
            args=(packet_queue, feature_queue, batch_size, stop_event, num_ai_workers)
        )
        processes.append(p)

    for _ in range(num_ai_workers):
        p = multiprocessing.Process(target=mp_ai_worker_fn,
                                     args=(feature_queue, model_path, scaler_path,
                                           result_queue, stop_event))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=120)

    elapsed = time.time() - start

    # Collect results
    all_latencies = []
    total_preds = 0
    while not result_queue.empty():
        preds, lats = result_queue.get()
        total_preds += preds
        all_latencies.extend(lats)

    result = {
        'throughput': total_preds / max(elapsed, 0.001),
        'total_time': elapsed,
        'predictions': total_preds,
    }
    if all_latencies:
        result.update({
            'avg_latency_ms': np.mean(all_latencies) * 1000,
            'p50_latency_ms': np.percentile(all_latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(all_latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(all_latencies, 99) * 1000,
        })
    else:
        result.update({
            'avg_latency_ms': 0, 'p50_latency_ms': 0,
            'p95_latency_ms': 0, 'p99_latency_ms': 0,
        })

    return result


def run_full_benchmark():
    """
    ╔════════════════════════════════════════════════════════╗
    ║  FULL BENCHMARK SUITE                                  ║
    ╠════════════════════════════════════════════════════════╣
    ║  1. Inference-only benchmark (batch sizes)             ║
    ║  2. Single-thread pipeline                            ║
    ║  3. Multi-thread pipeline (vary workers)              ║
    ║  4. Multi-process pipeline (vary workers)             ║
    ║  5. Scalability test (1 to N workers)                 ║
    ╚════════════════════════════════════════════════════════╝
    """
    print("╔" + "═" * 58 + "╗")
    print("║         BENCHMARK SUITE - NETWORK IDS                   ║")
    print("╚" + "═" * 58 + "╝")

    # Check model
    if not os.path.exists(config.MODEL_PATH):
        print("[!] Model chưa train. Running train_model.py first...")
        os.system(f"{sys.executable} train_model.py")

    model = IDSModel()
    model.load()
    scaler = joblib.load(config.SCALER_PATH)

    num_samples = config.BENCHMARK_NUM_PACKETS
    all_results = {}

    # ══════════════════════════════════════════════════════
    # TEST 1: Inference-only (cô lập AI performance)
    # ══════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEST 1: Inference-Only Benchmark")
    print("█" * 60)

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    inference_results = {}
    for bs in batch_sizes:
        r = benchmark_inference_only(model, scaler, min(num_samples, 10000), bs)
        inference_results[bs] = r
        print(f"  Batch {bs:>4}: {r['throughput']:>10,.0f} samples/sec | "
              f"Latency: {r['avg_latency_ms']:.3f} ms")
    all_results['inference_only'] = {str(k): v for k, v in inference_results.items()}

    # ══════════════════════════════════════════════════════
    # TEST 2: Single-Thread Pipeline
    # ══════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEST 2: Single-Thread Pipeline")
    print("█" * 60)

    r_st = benchmark_pipeline_single_thread(model, scaler, num_samples, config.BATCH_SIZE)
    print(f"  Throughput:  {r_st['throughput']:>10,.0f} preds/sec")
    print(f"  Avg Latency: {r_st['avg_latency_ms']:.2f} ms")
    print(f"  P95 Latency: {r_st['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency: {r_st['p99_latency_ms']:.2f} ms")
    all_results['single_thread'] = r_st

    # ══════════════════════════════════════════════════════
    # TEST 3: Multi-Thread Pipeline (vary workers)
    # ══════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEST 3: Multi-Thread Pipeline (Scalability)")
    print("█" * 60)

    mt_results = {}
    worker_counts = [1, 2, 4]
    for nw in worker_counts:
        if nw > multiprocessing.cpu_count():
            break
        r = benchmark_pipeline_multi_thread(
            model, scaler, num_samples, config.BATCH_SIZE,
            num_consumers=min(2, nw), num_ai_workers=nw
        )
        mt_results[nw] = r
        speedup = r['throughput'] / max(r_st['throughput'], 1)
        print(f"  Workers={nw}: {r['throughput']:>10,.0f} preds/sec | "
              f"Speedup: {speedup:.2f}x | "
              f"Avg Latency: {r['avg_latency_ms']:.2f} ms")
    all_results['multi_thread'] = {str(k): v for k, v in mt_results.items()}

    # ══════════════════════════════════════════════════════
    # TEST 4: Multi-Process Pipeline
    # ══════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEST 4: Multi-Process Pipeline (Scalability)")
    print("█" * 60)

    mp_results = {}
    for nw in worker_counts:
        if nw > multiprocessing.cpu_count():
            break
        r = benchmark_pipeline_multi_process(
            config.MODEL_PATH, config.SCALER_PATH,
            num_samples, config.BATCH_SIZE,
            num_consumers=min(2, nw), num_ai_workers=nw
        )
        mp_results[nw] = r
        speedup = r['throughput'] / max(r_st['throughput'], 1)
        print(f"  Workers={nw}: {r['throughput']:>10,.0f} preds/sec | "
              f"Speedup: {speedup:.2f}x | "
              f"Avg Latency: {r['avg_latency_ms']:.2f} ms")
    all_results['multi_process'] = {str(k): v for k, v in mp_results.items()}

    # ══════════════════════════════════════════════════════
    # TEST 5: Batch Size Impact
    # ══════════════════════════════════════════════════════
    print("\n" + "█" * 60)
    print("  TEST 5: Batch Size Impact (Multi-Thread, 2 workers)")
    print("█" * 60)

    bs_results = {}
    for bs in [4, 16, 32, 64, 128]:
        r = benchmark_pipeline_multi_thread(
            model, scaler, min(num_samples, 20000), bs,
            num_consumers=2, num_ai_workers=2
        )
        bs_results[bs] = r
        print(f"  Batch={bs:>4}: {r['throughput']:>10,.0f} preds/sec | "
              f"Avg Latency: {r['avg_latency_ms']:.2f} ms | "
              f"P95: {r['p95_latency_ms']:.2f} ms")
    all_results['batch_size_impact'] = {str(k): v for k, v in bs_results.items()}

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  TỔNG KẾT BENCHMARK")
    print("═" * 60)

    baseline = r_st['throughput']
    print(f"\n  {'Mode':<25} {'Throughput':>12} {'Speedup':>8} {'Avg Lat':>10}")
    print(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*10}")
    print(f"  {'Single-Thread':<25} {r_st['throughput']:>10,.0f}/s {'1.00x':>8} "
          f"{r_st['avg_latency_ms']:>8.2f}ms")

    for nw in sorted(mt_results.keys()):
        r = mt_results[nw]
        sp = r['throughput'] / max(baseline, 1)
        print(f"  {f'Multi-Thread ({nw}w)':<25} {r['throughput']:>10,.0f}/s "
              f"{sp:>7.2f}x {r['avg_latency_ms']:>8.2f}ms")

    for nw in sorted(mp_results.keys()):
        r = mp_results[nw]
        sp = r['throughput'] / max(baseline, 1)
        print(f"  {f'Multi-Process ({nw}w)':<25} {r['throughput']:>10,.0f}/s "
              f"{sp:>7.2f}x {r['avg_latency_ms']:>8.2f}ms")

    # Save
    results_path = os.path.join(config.RESULTS_DIR, "benchmark_results.json")

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n[+] Benchmark results saved: {results_path}")

    return all_results


if __name__ == "__main__":
    run_full_benchmark()
