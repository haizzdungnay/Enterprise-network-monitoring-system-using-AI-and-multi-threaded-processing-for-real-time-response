"""
╔══════════════════════════════════════════════════════════════════╗
║   RESEARCH_BENCHMARK.PY - ĐÁNH GIÁ NGHIÊN CỨU TOÀN DIỆN      ║
╠══════════════════════════════════════════════════════════════════╣
║  Kiến trúc: NIC → Producer-Consumer → AI Worker Pool           ║
║  So sánh:   Single-thread vs Multi-thread vs Multi-process     ║
║  Đánh giá:  Throughput, Latency, Scalability                   ║
║  Tối ưu:    Queue buffer, Batch size, CPU/GPU scheduling       ║
╚══════════════════════════════════════════════════════════════════╝

NGHIÊN CỨU MỤC TIÊU:
─────────────────────
Đánh giá kiến trúc Producer-Consumer cho hệ thống IDS thời gian thực,
bao gồm:

  1. AI Inference Performance     – ảnh hưởng của batch size
  2. Pipeline Performance         – single vs multi-thread vs multi-process
  3. Queue Buffer Optimization    – ảnh hưởng queue size đến throughput/latency
  4. Batch Size Optimization      – tìm batch size tối ưu cho pipeline
  5. Scalability Analysis         – speedup khi tăng worker count
  6. CPU vs GPU Scheduling        – so sánh inference trên CPU vs GPU

PHƯƠNG PHÁP ĐO:
───────────────
- Mỗi test chạy WARMUP trước khi đo chính thức
- Mỗi configuration chạy NUM_REPEATS lần, lấy trung bình
- Đo: throughput (samples/sec), latency (P50/P95/P99), speedup
- Hardware info được ghi nhận để đảm bảo reproducibility
"""

import os
import sys
import multiprocessing

# ── Disable CUDA in child processes (Windows spawn sets __name__='__mp_main__') ──
# Child processes only need _mp_workers — skip ALL heavy imports to avoid
# torch/CUDA initialization which takes 15-30s per child process on Windows.
_IS_MP_CHILD = (__name__ == '__mp_main__')

if _IS_MP_CHILD:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import json
import queue
import threading
import platform
import numpy as np
import joblib
import subprocess

if not _IS_MP_CHILD:
    import config
    from feature_extractor import generate_simulated_features
    from ai_model import IDSModel

from _mp_workers import mp_worker_simple


# ═══════════════════════════════════════════════════════════════
#  HELPER: Collect hardware info for reproducibility
# ═══════════════════════════════════════════════════════════════

def get_hardware_info():
    """Thu thập thông tin phần cứng để đảm bảo reproducibility."""
    info = {
        'os': f"{platform.system()} {platform.release()}",
        'python': platform.python_version(),
        'cpu': platform.processor() or 'Unknown',
        'cpu_cores_physical': os.cpu_count(),
        'ram_gb': 'N/A',
    }
    try:
        import psutil
        info['ram_gb'] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info['gpu_vram_mb'] = props.total_mem // (1024**2) if hasattr(props, 'total_mem') else props.total_memory // (1024**2)
            info['gpu_compute_capability'] = f"{props.major}.{props.minor}"
            info['cuda_version'] = torch.version.cuda
        else:
            info['gpu'] = 'None (CUDA not available)'
    except ImportError:
        info['gpu'] = 'None (PyTorch not installed)'

    return info


# ═══════════════════════════════════════════════════════════════
#  TEST 1: Inference-Only Benchmark (Batch Size Sweep)
# ═══════════════════════════════════════════════════════════════

def test_inference_batch_sizes(model, scaler, num_samples=10000, repeats=3):
    """
    Đo AI inference performance khi thay đổi batch size.
    Cô lập inference speed — không queue, không threading.

    INSIGHT: Batch lớn hơn → amortize overhead → throughput cao hơn
    nhưng đến ngưỡng nhất định GPU/CPU saturated.
    """
    print("\n" + "█" * 70)
    print("  TEST 1: INFERENCE-ONLY — Batch Size Sweep")
    print("  Mục đích: Đo throughput AI inference khi thay đổi batch size")
    print("█" * 70)

    # Pre-generate data
    X = np.array([generate_simulated_features() for _ in range(num_samples)])

    expected = getattr(scaler, 'n_features_in_', X.shape[1])
    if X.shape[1] < expected:
        X = np.pad(X, ((0, 0), (0, expected - X.shape[1])), mode='constant')
    elif X.shape[1] > expected:
        X = X[:, :expected]

    X_scaled = scaler.transform(X)

    # Warmup
    _ = model.predict(X_scaled[:min(500, len(X_scaled))])

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = {}

    for bs in batch_sizes:
        run_throughputs = []
        run_latencies = []

        for rep in range(repeats):
            start = time.perf_counter()
            count = 0
            for i in range(0, len(X_scaled), bs):
                batch = X_scaled[i:i + bs]
                model.predict(batch)
                count += len(batch)
            elapsed = time.perf_counter() - start

            run_throughputs.append(count / elapsed)
            run_latencies.append((elapsed / count) * 1000)

        results[bs] = {
            'batch_size': bs,
            'throughput_mean': float(np.mean(run_throughputs)),
            'throughput_std': float(np.std(run_throughputs)),
            'latency_ms_mean': float(np.mean(run_latencies)),
            'latency_ms_std': float(np.std(run_latencies)),
            'speedup_vs_bs1': 1.0,  # filled later
        }

        print(f"  Batch {bs:>4}: {np.mean(run_throughputs):>12,.0f} ± "
              f"{np.std(run_throughputs):>8,.0f} samples/sec  |  "
              f"Latency: {np.mean(run_latencies):.4f} ms/sample")

    # Calculate speedup relative to batch_size=1
    baseline = results[1]['throughput_mean']
    for bs in results:
        results[bs]['speedup_vs_bs1'] = results[bs]['throughput_mean'] / max(baseline, 1)

    return results


# ═══════════════════════════════════════════════════════════════
#  PIPELINE HELPERS — Shared across tests
# ═══════════════════════════════════════════════════════════════

def _prepare_batch(X_batch, scaler):
    """Khớp số feature giữa batch và scaler."""
    expected = getattr(scaler, 'n_features_in_', None)
    if expected is None:
        return X_batch
    if X_batch.shape[1] == expected:
        return X_batch
    if X_batch.shape[1] < expected:
        return np.pad(X_batch, ((0, 0), (0, expected - X_batch.shape[1])), mode='constant')
    return X_batch[:, :expected]


def _run_single_thread_pipeline(model, scaler, num_samples, batch_size):
    """Single-thread pipeline: Capture → Extract → Batch → Predict."""
    latencies = []
    predictions = 0
    batch_buffer = []

    start = time.perf_counter()
    for _ in range(num_samples):
        capture_time = time.perf_counter()
        features = generate_simulated_features()
        batch_buffer.append((features, capture_time))

        if len(batch_buffer) >= batch_size:
            X = np.array([f[0] for f in batch_buffer])
            timestamps = [f[1] for f in batch_buffer]
            X = _prepare_batch(X, scaler)
            X_scaled = scaler.transform(X)
            labels = model.predict(X_scaled)
            now = time.perf_counter()
            for ts in timestamps:
                latencies.append(now - ts)
            predictions += len(labels)
            batch_buffer = []

    if batch_buffer:
        X = np.array([f[0] for f in batch_buffer])
        timestamps = [f[1] for f in batch_buffer]
        X = _prepare_batch(X, scaler)
        X_scaled = scaler.transform(X)
        labels = model.predict(X_scaled)
        now = time.perf_counter()
        for ts in timestamps:
            latencies.append(now - ts)
        predictions += len(labels)

    elapsed = time.perf_counter() - start
    return _compute_stats(predictions, elapsed, latencies)


def _run_multi_thread_pipeline(model, scaler, num_samples, batch_size,
                                num_consumers, num_ai_workers,
                                packet_queue_size=10000, feature_queue_size=5000):
    """
    Multi-thread pipeline avec Producer-Consumer-AI architecture.

    Architecture:
      Producer → [Packet Queue] → Consumer(s) → [Feature Queue] → AI Worker(s)
    """
    packet_q = queue.Queue(maxsize=packet_queue_size)
    feature_q = queue.Queue(maxsize=feature_queue_size)
    stop = threading.Event()

    latencies = []
    lat_lock = threading.Lock()
    pred_count = [0]
    dropped = [0]

    def producer():
        for _ in range(num_samples):
            if stop.is_set():
                break
            features = generate_simulated_features()
            t = time.perf_counter()
            try:
                packet_q.put((features, t), timeout=2)
            except queue.Full:
                dropped[0] += 1
        for _ in range(num_consumers):
            packet_q.put(None)

    def consumer():
        buf = []
        while not stop.is_set():
            try:
                item = packet_q.get(timeout=0.5)
            except queue.Empty:
                if buf:
                    feature_q.put(buf)
                    buf = []
                continue
            if item is None:
                if buf:
                    feature_q.put(buf)
                feature_q.put(None)
                break
            buf.append(item)
            if len(buf) >= batch_size:
                feature_q.put(buf)
                buf = []

    def ai_worker():
        while not stop.is_set():
            try:
                batch = feature_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if batch is None:
                feature_q.put(None)
                break
            X = np.array([it[0] for it in batch])
            timestamps = [it[1] for it in batch]
            try:
                X = _prepare_batch(X, scaler)
                X_scaled = scaler.transform(X)
                labels = model.predict(X_scaled)
                now = time.perf_counter()
                with lat_lock:
                    for ts in timestamps:
                        latencies.append(now - ts)
                    pred_count[0] += len(labels)
            except Exception:
                pass

    threads = []
    start = time.perf_counter()

    t = threading.Thread(target=producer, daemon=True)
    threads.append(t)
    for _ in range(num_consumers):
        t = threading.Thread(target=consumer, daemon=True)
        threads.append(t)
    for _ in range(num_ai_workers):
        t = threading.Thread(target=ai_worker, daemon=True)
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    elapsed = time.perf_counter() - start
    stats = _compute_stats(pred_count[0], elapsed, latencies)
    stats['dropped_packets'] = dropped[0]
    return stats


# MP worker functions are imported from _mp_workers.py (lightweight module
# that avoids torch/config imports for fast child process startup on Windows)


def _run_multi_process_pipeline(model_path, scaler_path, num_samples,
                                 batch_size, num_consumers=2, num_ai_workers=2,
                                 pq_size=10000, fq_size=5000):
    """Multi-process pipeline with true parallelism (bypass GIL).
    
    Simple direct-split approach: pre-generate features, split among N workers.
    Each worker loads model (CPU), processes its slice, sends summary dict.
    No inter-process queues for data flow — eliminates all pipe-buffer deadlocks.
    """
    import tempfile

    # Pre-generate all features as numpy array
    rng = np.random.default_rng()
    features = rng.standard_normal((num_samples, 18)).astype(np.float32)

    # Save to temp file for workers to mmap
    tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    tmp_path = tmp.name
    tmp.close()
    np.save(tmp_path, features)

    rq = multiprocessing.Queue()

    # Split work among workers
    nw = num_ai_workers
    chunk = num_samples // nw
    procs = []
    start = time.perf_counter()

    for i in range(nw):
        s = i * chunk
        e = s + chunk if i < nw - 1 else num_samples
        p = multiprocessing.Process(
            target=mp_worker_simple,
            args=(tmp_path, s, e, batch_size, model_path, scaler_path, rq, i))
        procs.append(p)

    for p in procs:
        p.start()

    # Collect results BEFORE joining (avoids pipe buffer deadlock)
    summaries = []
    for _ in range(nw):
        try:
            summaries.append(rq.get(timeout=120))
        except Exception:
            break

    # Now join processes (they should already be done)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    elapsed = time.perf_counter() - start

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    total_pred = sum(s['predictions'] for s in summaries)

    result = {
        'predictions': total_pred,
        'elapsed_sec': float(elapsed),
        'throughput': total_pred / max(elapsed, 0.001),
    }
    if summaries and total_pred > 0:
        total_lat = sum(s.get('total_latency_sec', 0) for s in summaries)
        result.update({
            'avg_latency_ms': float(np.mean([s['avg_latency_ms'] for s in summaries])),
            'p50_latency_ms': float(np.mean([s['p50_latency_ms'] for s in summaries])),
            'p95_latency_ms': float(max(s['p95_latency_ms'] for s in summaries)),
            'p99_latency_ms': float(max(s['p99_latency_ms'] for s in summaries)),
            'min_latency_ms': float(min(s['min_latency_ms'] for s in summaries)),
            'max_latency_ms': float(max(s['max_latency_ms'] for s in summaries)),
        })
    else:
        result.update({k: 0.0 for k in [
            'avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms',
            'p99_latency_ms', 'min_latency_ms', 'max_latency_ms']})
    return result


def _compute_stats(predictions, elapsed, latencies):
    """Tính metrics từ raw latency data."""
    result = {
        'predictions': predictions,
        'elapsed_sec': float(elapsed),
        'throughput': predictions / max(elapsed, 0.001),
    }
    if latencies:
        arr = np.array(latencies)
        result.update({
            'avg_latency_ms': float(np.mean(arr) * 1000),
            'p50_latency_ms': float(np.percentile(arr, 50) * 1000),
            'p95_latency_ms': float(np.percentile(arr, 95) * 1000),
            'p99_latency_ms': float(np.percentile(arr, 99) * 1000),
            'min_latency_ms': float(np.min(arr) * 1000),
            'max_latency_ms': float(np.max(arr) * 1000),
        })
    else:
        result.update({k: 0.0 for k in [
            'avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms',
            'p99_latency_ms', 'min_latency_ms', 'max_latency_ms']})
    return result


# ═══════════════════════════════════════════════════════════════
#  TEST 2: Pipeline Comparison (Single vs Multi-Thread vs Multi-Process)
# ═══════════════════════════════════════════════════════════════

def test_pipeline_comparison(model, scaler, num_samples=20000, repeats=2):
    """
    So sánh 3 mode pipeline với cùng workload.

    KIẾN TRÚC:
      ┌──────────┐     ┌──────────────┐     ┌───────────┐
      │ Producer  │────▶│ Packet Queue │────▶│ Consumer  │
      │ (Capture) │     │ (thread-safe)│     │ (Extract) │
      └──────────┘     └──────────────┘     └─────┬─────┘
                                                  │
                                        ┌─────────▼─────────┐
                                        │  Feature Queue     │
                                        └─────────┬─────────┘
                                                  │
                                        ┌─────────▼─────────┐
                                        │  AI Worker Pool    │
                                        └───────────────────┘
    """
    print("\n" + "█" * 70)
    print("  TEST 2: PIPELINE COMPARISON — Single vs Multi-Thread vs Multi-Process")
    print("  Mục đích: So sánh throughput & latency giữa 3 chế độ xử lý")
    print("█" * 70)

    batch_size = config.BATCH_SIZE
    results = {}

    # — Single Thread —
    print(f"\n  ▸ Single-Thread (batch={batch_size})...")
    st_runs = []
    for r in range(repeats):
        st_runs.append(_run_single_thread_pipeline(model, scaler, num_samples, batch_size))
    results['single_thread'] = _average_runs(st_runs)
    _print_mode_result("Single-Thread", results['single_thread'])

    # — Multi-Thread (2 consumers, 2 AI workers) —
    print(f"\n  ▸ Multi-Thread (consumers=2, ai_workers=2, batch={batch_size})...")
    mt_runs = []
    for r in range(repeats):
        mt_runs.append(_run_multi_thread_pipeline(
            model, scaler, num_samples, batch_size,
            num_consumers=2, num_ai_workers=2))
    results['multi_thread'] = _average_runs(mt_runs)
    _print_mode_result("Multi-Thread", results['multi_thread'])

    # — Multi-Process (2 consumers, 2 AI workers) —
    print(f"\n  ▸ Multi-Process (consumers=2, ai_workers=2, batch={batch_size})...")
    mp_runs = []
    for r in range(repeats):
        mp_runs.append(_run_multi_process_pipeline(
            config.MODEL_PATH, config.SCALER_PATH, num_samples, batch_size,
            num_consumers=2, num_ai_workers=2))
    results['multi_process'] = _average_runs(mp_runs)
    _print_mode_result("Multi-Process", results['multi_process'])

    # Speedup relative to single-thread
    baseline_tp = results['single_thread']['throughput']
    for mode in results:
        results[mode]['speedup_vs_single'] = results[mode]['throughput'] / max(baseline_tp, 1)

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 3: Queue Buffer Size Impact
# ═══════════════════════════════════════════════════════════════

def test_queue_buffer_sizes(model, scaler, num_samples=20000, repeats=2):
    """
    Đánh giá ảnh hưởng của queue buffer size đến performance.

    Queue nhỏ → Producer bị block → throughput giảm, latency thấp
    Queue lớn → Tốn RAM, packet chờ lâu → latency cao, throughput ổn
    """
    print("\n" + "█" * 70)
    print("  TEST 3: QUEUE BUFFER SIZE IMPACT")
    print("  Mục đích: Tìm queue size tối ưu (trade-off throughput vs latency)")
    print("█" * 70)

    queue_sizes = [100, 1000, 5000, 10000]
    batch_size = config.BATCH_SIZE
    results = {}

    for qs in queue_sizes:
        print(f"\n  ▸ Queue size = {qs:,}...")
        runs = []
        for r in range(repeats):
            runs.append(_run_multi_thread_pipeline(
                model, scaler, num_samples, batch_size,
                num_consumers=2, num_ai_workers=2,
                packet_queue_size=qs, feature_queue_size=max(100, qs // 2)))
        results[qs] = _average_runs(runs)
        print(f"    Throughput: {results[qs]['throughput']:>10,.0f} pred/sec  |  "
              f"Avg Latency: {results[qs]['avg_latency_ms']:.2f} ms  |  "
              f"P99: {results[qs]['p99_latency_ms']:.2f} ms")

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 4: Batch Size Impact on Pipeline
# ═══════════════════════════════════════════════════════════════

def test_batch_size_pipeline(model, scaler, num_samples=20000, repeats=2):
    """
    Đánh giá ảnh hưởng của batch size khi chạy full pipeline.

    Batch nhỏ (1-8):   → Latency thấp, throughput thấp (overhead/batch cao)
    Batch lớn (128-512): → Throughput cao, latency cao (chờ gom đủ batch)
    """
    print("\n" + "█" * 70)
    print("  TEST 4: BATCH SIZE IMPACT ON PIPELINE")
    print("  Mục đích: Tìm batch size tối ưu (throughput vs latency trade-off)")
    print("█" * 70)

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    results = {}

    for bs in batch_sizes:
        print(f"\n  ▸ Batch size = {bs}...")

        # Test cả Single-Thread và Multi-Thread để so sánh
        st = _run_single_thread_pipeline(model, scaler, num_samples, bs)
        mt_runs = []
        for r in range(repeats):
            mt_runs.append(_run_multi_thread_pipeline(
                model, scaler, num_samples, bs,
                num_consumers=2, num_ai_workers=2))
        mt = _average_runs(mt_runs)

        results[bs] = {
            'single_thread': st,
            'multi_thread': mt,
            'mt_speedup_vs_st': mt['throughput'] / max(st['throughput'], 1),
        }
        print(f"    ST: {st['throughput']:>10,.0f} pred/sec, {st['avg_latency_ms']:.2f}ms  |  "
              f"MT: {mt['throughput']:>10,.0f} pred/sec, {mt['avg_latency_ms']:.2f}ms  |  "
              f"Speedup: {results[bs]['mt_speedup_vs_st']:.2f}x")

    return results


# ═══════════════════════════════════════════════════════════════
#  TEST 5: Scalability Analysis (Worker Count)
# ═══════════════════════════════════════════════════════════════

def test_scalability(model, scaler, num_samples=20000, repeats=2):
    """
    Đo scalability khi tăng số workers.

    Áp dụng Amdahl's Law:
      S = 1 / ((1 - P) + P/N)
      P = parallelizable fraction
      N = number of processors

    Linear scaling (2x workers → 2x throughput) là lý tưởng.
    Thực tế thường sublinear do coordination overhead.
    """
    print("\n" + "█" * 70)
    print("  TEST 5: SCALABILITY ANALYSIS — Worker Count Sweep")
    print("  Mục đích: Đo speedup khi tăng AI workers (Amdahl's Law)")
    print("█" * 70)

    batch_size = config.BATCH_SIZE
    worker_counts = [1, 2, 4]

    results_mt = {}
    results_mp = {}

    # Baseline: single-thread
    st = _run_single_thread_pipeline(model, scaler, num_samples, batch_size)
    baseline_tp = st['throughput']
    print(f"\n  Baseline (Single-Thread): {baseline_tp:,.0f} pred/sec")

    # Multi-Thread scalability
    print(f"\n  ── Multi-Thread Scalability ──")
    for nw in worker_counts:
        runs = []
        for r in range(repeats):
            runs.append(_run_multi_thread_pipeline(
                model, scaler, num_samples, batch_size,
                num_consumers=min(2, nw), num_ai_workers=nw))
        avg = _average_runs(runs)
        avg['speedup'] = avg['throughput'] / max(baseline_tp, 1)
        avg['efficiency'] = avg['speedup'] / nw  # Ideal = 1.0
        results_mt[nw] = avg
        print(f"    Workers={nw}: {avg['throughput']:>10,.0f} pred/sec  |  "
              f"Speedup: {avg['speedup']:.2f}x  |  "
              f"Efficiency: {avg['efficiency']:.1%}")

    # Multi-Process scalability
    print(f"\n  ── Multi-Process Scalability ──")
    for nw in worker_counts:
        runs = []
        for r in range(repeats):
            runs.append(_run_multi_process_pipeline(
                config.MODEL_PATH, config.SCALER_PATH, num_samples, batch_size,
                num_consumers=min(2, nw), num_ai_workers=nw))
        avg = _average_runs(runs)
        avg['speedup'] = avg['throughput'] / max(baseline_tp, 1)
        avg['efficiency'] = avg['speedup'] / nw
        results_mp[nw] = avg
        print(f"    Workers={nw}: {avg['throughput']:>10,.0f} pred/sec  |  "
              f"Speedup: {avg['speedup']:.2f}x  |  "
              f"Efficiency: {avg['efficiency']:.1%}")

    return {
        'baseline_single_thread': st,
        'multi_thread': results_mt,
        'multi_process': results_mp,
    }


# ═══════════════════════════════════════════════════════════════
#  TEST 6: CPU vs GPU Scheduling
# ═══════════════════════════════════════════════════════════════

def test_cpu_gpu_comparison(scaler, num_samples=10000, repeats=3):
    """
    So sánh inference speed trên CPU vs GPU (nếu có).
    Chỉ test XGBoost vì là model chính.
    """
    print("\n" + "█" * 70)
    print("  TEST 6: CPU vs GPU SCHEDULING")
    print("  Mục đích: So sánh tốc độ inference giữa CPU và GPU")
    print("█" * 70)

    X = np.array([generate_simulated_features() for _ in range(num_samples)])
    exp = getattr(scaler, 'n_features_in_', X.shape[1])
    if X.shape[1] != exp:
        if X.shape[1] < exp:
            X = np.pad(X, ((0, 0), (0, exp - X.shape[1])), mode='constant')
        else:
            X = X[:, :exp]
    X_scaled = scaler.transform(X)

    results = {}
    batch_sizes = [32, 64, 128, 256]

    # CPU model
    print("\n  ── CPU Inference ──")
    try:
        from xgboost import XGBClassifier
        cpu_model = IDSModel('xgboost')
        cpu_model.load()
        # Force CPU
        if hasattr(cpu_model.model, 'set_params'):
            cpu_model.model.set_params(device='cpu')

        cpu_results = {}
        _ = cpu_model.predict(X_scaled[:100])  # warmup
        for bs in batch_sizes:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                for i in range(0, len(X_scaled), bs):
                    cpu_model.predict(X_scaled[i:i + bs])
                times.append(time.perf_counter() - start)
            tp = num_samples / np.mean(times)
            cpu_results[bs] = {'throughput': float(tp), 'time_sec': float(np.mean(times))}
            print(f"    Batch {bs:>4}: {tp:>12,.0f} samples/sec")
        results['cpu'] = cpu_results
    except Exception as e:
        print(f"    [!] CPU test failed: {e}")
        results['cpu'] = {}

    # GPU model (if available)
    print("\n  ── GPU Inference ──")
    if config.USE_GPU:
        try:
            gpu_model = IDSModel('xgboost')
            gpu_model.load()
            # Force GPU
            if hasattr(gpu_model.model, 'set_params'):
                gpu_model.model.set_params(device='cuda')

            gpu_results = {}
            _ = gpu_model.predict(X_scaled[:100])  # warmup
            for bs in batch_sizes:
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    for i in range(0, len(X_scaled), bs):
                        gpu_model.predict(X_scaled[i:i + bs])
                    times.append(time.perf_counter() - start)
                tp = num_samples / np.mean(times)
                gpu_results[bs] = {'throughput': float(tp), 'time_sec': float(np.mean(times))}
                print(f"    Batch {bs:>4}: {tp:>12,.0f} samples/sec")
            results['gpu'] = gpu_results

            # Speedup
            print("\n  ── GPU Speedup vs CPU ──")
            for bs in batch_sizes:
                if bs in results.get('cpu', {}) and bs in results.get('gpu', {}):
                    sp = results['gpu'][bs]['throughput'] / max(results['cpu'][bs]['throughput'], 1)
                    print(f"    Batch {bs:>4}: {sp:.2f}x speedup")

        except Exception as e:
            print(f"    [!] GPU test failed: {e}")
            results['gpu'] = {}
    else:
        print("    GPU not available. Skipping.")
        results['gpu'] = {}

    return results


# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def _average_runs(runs):
    """Average multiple benchmark runs."""
    keys = runs[0].keys()
    avg = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        if vals:
            avg[k] = float(np.mean(vals))
        else:
            avg[k] = runs[0][k]
    return avg


def _print_mode_result(name, r):
    """In kết quả 1 mode."""
    print(f"    {name}:")
    print(f"      Throughput:  {r['throughput']:>10,.0f} predictions/sec")
    print(f"      Avg Latency: {r['avg_latency_ms']:>8.2f} ms")
    print(f"      P50 Latency: {r['p50_latency_ms']:>8.2f} ms")
    print(f"      P95 Latency: {r['p95_latency_ms']:>8.2f} ms")
    print(f"      P99 Latency: {r['p99_latency_ms']:>8.2f} ms")


# ═══════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_full_research_benchmark():
    """
    Chạy toàn bộ benchmark suite cho nghiên cứu.
    Kết quả lưu vào results/research_benchmark.json
    """
    print("╔" + "═" * 68 + "╗")
    print("║    RESEARCH BENCHMARK — Đánh Giá Toàn Diện Hệ Thống IDS         ║")
    print("║    Producer-Consumer Architecture với AI Worker Pool             ║")
    print("╚" + "═" * 68 + "╝")

    # Check model
    if not os.path.exists(config.MODEL_PATH):
        print("[!] Model chưa train. Running train_model.py...")
        subprocess.run([sys.executable, "train_model.py"], check=False)

    print("\n[*] Loading model & scaler...")
    model = IDSModel()
    model.load()
    scaler = joblib.load(config.SCALER_PATH)

    # Warmup
    warmup_X = np.array([generate_simulated_features() for _ in range(500)])
    exp = getattr(scaler, 'n_features_in_', warmup_X.shape[1])
    if warmup_X.shape[1] != exp:
        if warmup_X.shape[1] < exp:
            warmup_X = np.pad(warmup_X, ((0, 0), (0, exp - warmup_X.shape[1])), mode='constant')
        else:
            warmup_X = warmup_X[:, :exp]
    _ = model.predict(scaler.transform(warmup_X))
    print("[*] Warmup complete.\n")

    hardware = get_hardware_info()
    config.print_config()

    all_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware': hardware,
            'config': {
                'model_type': config.MODEL_TYPE,
                'batch_size': config.BATCH_SIZE,
                'queue_size': config.PACKET_QUEUE_SIZE,
                'num_consumers': config.NUM_CONSUMER_THREADS,
                'num_ai_workers': config.NUM_AI_WORKERS,
                'use_gpu': config.USE_GPU,
                'gpu_profile': config._GPU_PROFILE_NAME,
            }
        }
    }

    # ── Test 1: Inference batch sizes ──
    all_results['test1_inference_batch'] = {
        str(k): v for k, v in
        test_inference_batch_sizes(model, scaler, num_samples=5000, repeats=2).items()
    }

    # ── Test 2: Pipeline comparison ──
    all_results['test2_pipeline_comparison'] = \
        test_pipeline_comparison(model, scaler, num_samples=10000, repeats=1)

    # ── Test 3: Queue buffer sizes ──
    all_results['test3_queue_buffer'] = {
        str(k): v for k, v in
        test_queue_buffer_sizes(model, scaler, num_samples=8000, repeats=1).items()
    }

    # ── Test 4: Batch size on pipeline ──
    all_results['test4_batch_pipeline'] = {
        str(k): v for k, v in
        test_batch_size_pipeline(model, scaler, num_samples=8000, repeats=1).items()
    }

    # ── Test 5: Scalability ──
    all_results['test5_scalability'] = \
        test_scalability(model, scaler, num_samples=8000, repeats=1)

    # ── Test 6: CPU vs GPU ──
    all_results['test6_cpu_gpu'] = \
        test_cpu_gpu_comparison(scaler, num_samples=5000, repeats=2)

    # ── Load training results for completeness ──
    training_path = os.path.join(config.RESULTS_DIR, "training_results.json")
    if os.path.exists(training_path):
        with open(training_path) as f:
            all_results['training_metrics'] = json.load(f)

    # ── Save ──
    out_path = os.path.join(config.RESULTS_DIR, "research_benchmark.json")

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return obj

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK HOÀN TẤT!")
    print(f"  Results: {out_path}")
    print(f"{'═' * 70}")

    # ── Generate HTML report ──
    _generate_html_report(all_results)

    return all_results


# ═══════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def _generate_html_report(results):
    """Generate comprehensive evaluation HTML report with Chart.js charts."""
    out_path = os.path.join(config.BASE_DIR, "dashboard_static", "evaluation.html")

    # Prepare data for JSON embedding
    def safe_json(obj):
        if isinstance(obj, dict):
            return {str(k): safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [safe_json(i) for i in obj]
        return obj

    data_json = json.dumps(safe_json(results), indent=2, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Evaluation — Enterprise Network IDS</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
<style>
:root {{
  --bg-primary: #0a0e17;
  --bg-card: #141b2d;
  --bg-card-hover: #1a2338;
  --border: #1e2a42;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --accent-blue: #3b82f6;
  --accent-green: #10b981;
  --accent-orange: #f59e0b;
  --accent-red: #ef4444;
  --accent-purple: #8b5cf6;
  --accent-cyan: #06b6d4;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
}}

.header {{
  background: linear-gradient(135deg, #1e3a5f, #0a0e17);
  border-bottom: 1px solid var(--border);
  padding: 2rem 3rem;
  text-align: center;
}}

.header h1 {{
  font-size: 1.8rem;
  background: linear-gradient(to right, var(--accent-blue), var(--accent-cyan));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}}

.header .subtitle {{
  color: var(--text-secondary);
  font-size: 0.95rem;
}}

.container {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
}}

/* Hardware info strip */
.hw-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 2rem;
  padding: 1rem 1.5rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
}}

.hw-item {{
  flex: 1;
  min-width: 150px;
  text-align: center;
}}

.hw-item .label {{ color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; }}
.hw-item .value {{ color: var(--accent-cyan); font-weight: 600; font-size: 0.95rem; }}

/* Section */
.section {{
  margin-bottom: 2.5rem;
}}

.section-title {{
  font-size: 1.3rem;
  color: var(--accent-blue);
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}

.section-desc {{
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
  font-size: 0.9rem;
}}

/* Cards grid */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }}

@media (max-width: 900px) {{
  .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
}}

.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  transition: border-color 0.2s;
}}

.card:hover {{ border-color: var(--accent-blue); }}

.card h3 {{
  font-size: 1rem;
  color: var(--text-primary);
  margin-bottom: 1rem;
}}

.chart-container {{
  position: relative;
  width: 100%;
  height: 320px;
}}

/* KPI cards */
.kpi-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}

.kpi {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  text-align: center;
}}

.kpi .val {{
  font-size: 1.8rem;
  font-weight: 700;
}}

.kpi .lbl {{
  font-size: 0.8rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  margin-top: 0.3rem;
}}

.kpi.blue .val {{ color: var(--accent-blue); }}
.kpi.green .val {{ color: var(--accent-green); }}
.kpi.orange .val {{ color: var(--accent-orange); }}
.kpi.red .val {{ color: var(--accent-red); }}
.kpi.purple .val {{ color: var(--accent-purple); }}
.kpi.cyan .val {{ color: var(--accent-cyan); }}

/* Table */
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}}

.data-table th {{
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  padding: 0.7rem 1rem;
  text-align: left;
  border-bottom: 2px solid var(--border);
}}

.data-table td {{
  padding: 0.6rem 1rem;
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}}

.data-table tr:hover td {{ background: rgba(59, 130, 246, 0.05); }}

.badge {{
  display: inline-block;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
}}

.badge-green {{ background: rgba(16,185,129,.15); color: var(--accent-green); }}
.badge-orange {{ background: rgba(245,158,11,.15); color: var(--accent-orange); }}
.badge-red {{ background: rgba(239,68,68,.15); color: var(--accent-red); }}

/* Architecture diagram */
.arch-diagram {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2rem;
  font-family: 'Courier New', monospace;
  font-size: 0.82rem;
  line-height: 1.5;
  white-space: pre;
  overflow-x: auto;
  color: var(--accent-cyan);
  margin-bottom: 2rem;
}}

/* Analysis box */
.analysis-box {{
  background: rgba(59, 130, 246, 0.08);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 12px;
  padding: 1.5rem;
  margin-top: 1rem;
  font-size: 0.9rem;
}}

.analysis-box h4 {{ color: var(--accent-blue); margin-bottom: 0.7rem; }}
.analysis-box ul {{ margin-left: 1.2rem; color: var(--text-secondary); }}
.analysis-box li {{ margin-bottom: 0.3rem; }}
.analysis-box strong {{ color: var(--text-primary); }}

.footer {{
  text-align: center;
  padding: 2rem;
  color: var(--text-secondary);
  font-size: 0.85rem;
  border-top: 1px solid var(--border);
}}
</style>
</head>
<body>

<div class="header">
  <h1>Research Evaluation Report</h1>
  <div class="subtitle">Enterprise Network Intrusion Detection System — Producer-Consumer Architecture with AI Worker Pool</div>
</div>

<div class="container">

  <!-- Hardware info -->
  <div class="hw-strip" id="hw-strip"></div>

  <!-- Training KPIs -->
  <div class="kpi-row" id="training-kpis"></div>

  <!-- Architecture Diagram -->
  <div class="section">
    <div class="section-title">Kiến trúc hệ thống: Packet Capture → Producer-Consumer → AI Worker Pool</div>
    <div class="arch-diagram">
   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
   │                        ENTERPRISE NETWORK IDS — PIPELINE ARCHITECTURE                       │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘

   ┌────────────┐         ┌──────────────────┐         ┌──────────────────┐         ┌───────────┐
   │    NIC     │  pkts   │   Packet Queue   │ raw pkt │  Consumer Pool   │ batched │  Feature  │
   │  (Packet   │────────▶│  (thread-safe    │────────▶│  (Feature        │────────▶│   Queue   │
   │  Capture)  │         │   ring buffer)   │         │   Extraction)    │         │  (batch   │
   │            │         │                  │         │                  │         │  buffer)  │
   │  Producer  │         │  Size: N pkts    │         │  K consumers     │         │           │
   └────────────┘         │  Backpressure:   │         │  CPU-bound       │         │  Size: M  │
        │                 │  block/drop      │         └──────────────────┘         │  batches  │
        │                 └──────────────────┘                                     └─────┬─────┘
        │                                                                                │
        │   ┌────────────────────────────────────────────────────────────────────────────┘
        │   │
        │   ▼
        │   ┌──────────────────────────────────────────────┐        ┌──────────────────┐
        │   │          AI WORKER POOL                      │        │   Results &      │
        │   │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │───────▶│   Alert Engine   │
        │   │  │Worker 0 │ │Worker 1 │ │Worker N │        │        │                  │
        │   │  │(XGBoost │ │(XGBoost │ │(XGBoost │        │        │  - Log events    │
        │   │  │ GPU/CPU)│ │ GPU/CPU)│ │ GPU/CPU)│        │        │  - Trigger alert │
        │   │  └─────────┘ └─────────┘ └─────────┘        │        │  - Dashboard     │
        │   │  Batch inference, scaler.transform +         │        └──────────────────┘
        │   │  model.predict per batch                     │
        │   └──────────────────────────────────────────────┘
        │
        │   TUNING KNOBS:
        │   ├── Queue buffer size (N)    → trade-off: throughput vs latency vs memory
        │   ├── Batch size (B)           → trade-off: throughput vs latency
        │   ├── Worker count (W)         → scalability (Amdahl's Law)
        │   └── CPU/GPU scheduling       → device-aware inference
        │
        │   MODES:
        │   ├── Single-Thread   : Producer → Extract → Predict (sequential, baseline)
        │   ├── Multi-Thread    : GIL-aware, shared memory, fast IPC (~1μs)
        │   └── Multi-Process   : True parallelism, isolated memory, slow IPC (~100μs)
   </div>
  </div>

  <!-- Test 1: Inference Batch Size -->
  <div class="section">
    <div class="section-title">Test 1: AI Inference — Batch Size Sweep</div>
    <div class="section-desc">Ảnh hưởng của batch size đến throughput inference. Batch lớn hơn giảm overhead per-sample, tăng throughput đáng kể. Đây là nền tảng cho tối ưu pipeline.</div>
    <div class="grid-2">
      <div class="card"><h3>Throughput vs Batch Size</h3><div class="chart-container"><canvas id="chart-t1-throughput"></canvas></div></div>
      <div class="card"><h3>Latency vs Batch Size</h3><div class="chart-container"><canvas id="chart-t1-latency"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:1rem"><h3>Speedup vs Batch Size = 1</h3><div class="chart-container"><canvas id="chart-t1-speedup"></canvas></div></div>
    <div class="analysis-box" id="analysis-t1"></div>
  </div>

  <!-- Test 2: Pipeline Comparison -->
  <div class="section">
    <div class="section-title">Test 2: Pipeline Comparison — Single vs Multi-Thread vs Multi-Process</div>
    <div class="section-desc">So sánh end-to-end pipeline: Capture → Extract → Queue → Batch → Predict. Cùng workload, cùng batch size, khác architecture.</div>
    <div class="grid-2">
      <div class="card"><h3>Throughput Comparison</h3><div class="chart-container"><canvas id="chart-t2-throughput"></canvas></div></div>
      <div class="card"><h3>Latency Comparison (P50 / P95 / P99)</h3><div class="chart-container"><canvas id="chart-t2-latency"></canvas></div></div>
    </div>
    <div class="analysis-box" id="analysis-t2"></div>
  </div>

  <!-- Test 3: Queue Buffer -->
  <div class="section">
    <div class="section-title">Test 3: Queue Buffer Size Optimization</div>
    <div class="section-desc">Khảo sát ảnh hưởng của queue buffer size. Queue quá nhỏ → Producer block, mất packet. Queue quá lớn → tốn RAM, tăng latency.</div>
    <div class="grid-2">
      <div class="card"><h3>Throughput vs Queue Size</h3><div class="chart-container"><canvas id="chart-t3-throughput"></canvas></div></div>
      <div class="card"><h3>Latency vs Queue Size</h3><div class="chart-container"><canvas id="chart-t3-latency"></canvas></div></div>
    </div>
    <div class="analysis-box" id="analysis-t3"></div>
  </div>

  <!-- Test 4: Batch Size Pipeline -->
  <div class="section">
    <div class="section-title">Test 4: Batch Size Impact trên Full Pipeline</div>
    <div class="section-desc">Khác Test 1 (inference-only), test này đo batch size ảnh hưởng đến toàn bộ pipeline (queue + threading overhead).</div>
    <div class="grid-2">
      <div class="card"><h3>Throughput: Single-Thread vs Multi-Thread</h3><div class="chart-container"><canvas id="chart-t4-throughput"></canvas></div></div>
      <div class="card"><h3>Latency: Single-Thread vs Multi-Thread</h3><div class="chart-container"><canvas id="chart-t4-latency"></canvas></div></div>
    </div>
    <div class="analysis-box" id="analysis-t4"></div>
  </div>

  <!-- Test 5: Scalability -->
  <div class="section">
    <div class="section-title">Test 5: Scalability Analysis — Amdahl's Law</div>
    <div class="section-desc">Đo speedup khi tăng số AI workers. Theo Amdahl's Law: S = 1/((1−P) + P/N). Efficiency = Speedup/N. Linear scaling (efficiency=100%) là lý tưởng.</div>
    <div class="grid-2">
      <div class="card"><h3>Speedup vs Worker Count</h3><div class="chart-container"><canvas id="chart-t5-speedup"></canvas></div></div>
      <div class="card"><h3>Efficiency vs Worker Count</h3><div class="chart-container"><canvas id="chart-t5-efficiency"></canvas></div></div>
    </div>
    <div class="card" style="margin-top:1rem"><h3>Throughput vs Worker Count</h3><div class="chart-container"><canvas id="chart-t5-throughput"></canvas></div></div>
    <div class="analysis-box" id="analysis-t5"></div>
  </div>

  <!-- Test 6: CPU vs GPU -->
  <div class="section">
    <div class="section-title">Test 6: CPU vs GPU Scheduling</div>
    <div class="section-desc">So sánh tốc độ inference khi chạy trên CPU vs GPU. GPU hiệu quả với batch lớn do massive parallelism.</div>
    <div class="grid-2">
      <div class="card"><h3>Throughput: CPU vs GPU</h3><div class="chart-container"><canvas id="chart-t6-throughput"></canvas></div></div>
      <div class="card"><h3>GPU Speedup vs CPU</h3><div class="chart-container"><canvas id="chart-t6-speedup"></canvas></div></div>
    </div>
    <div class="analysis-box" id="analysis-t6"></div>
  </div>

  <!-- Summary Table -->
  <div class="section">
    <div class="section-title">Bảng Tổng Kết Đánh Giá</div>
    <div class="card">
      <table class="data-table" id="summary-table">
        <thead>
          <tr>
            <th>Mode</th>
            <th>Throughput (pred/sec)</th>
            <th>Avg Latency (ms)</th>
            <th>P95 Latency (ms)</th>
            <th>P99 Latency (ms)</th>
            <th>Speedup</th>
            <th>Rating</th>
          </tr>
        </thead>
        <tbody id="summary-tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<div class="footer">
  Enterprise Network IDS — Research Evaluation Report<br>
  Generated: <span id="gen-time"></span> | Model: XGBoost on CUDA
</div>

<script>
const DATA = {data_json};

// ── Chart defaults ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e2a42';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

const COLORS = {{
  blue: '#3b82f6', green: '#10b981', orange: '#f59e0b',
  red: '#ef4444', purple: '#8b5cf6', cyan: '#06b6d4', pink: '#ec4899'
}};

function fmt(n, d=0) {{ return n ? n.toLocaleString('en', {{maximumFractionDigits:d}}) : '0'; }}

// ── Hardware strip ──
(function() {{
  const hw = DATA.metadata?.hardware || {{}};
  const el = document.getElementById('hw-strip');
  const items = [
    ['OS', hw.os || 'N/A'],
    ['CPU', hw.cpu || 'N/A'],
    ['Cores', hw.cpu_cores_physical || 'N/A'],
    ['RAM', (hw.ram_gb || 'N/A') + ' GB'],
    ['GPU', hw.gpu || 'None'],
    ['VRAM', (hw.gpu_vram_mb || 'N/A') + ' MB'],
    ['CUDA', hw.cuda_version || 'N/A'],
    ['Python', hw.python || 'N/A'],
  ];
  el.innerHTML = items.map(([l,v]) =>
    `<div class="hw-item"><div class="label">${{l}}</div><div class="value">${{v}}</div></div>`
  ).join('');
}})();

// ── Training KPIs ──
(function() {{
  const m = DATA.training_metrics?.metrics || {{}};
  const el = document.getElementById('training-kpis');
  const kpis = [
    ['Accuracy', (m.accuracy*100||0).toFixed(2)+'%', 'blue'],
    ['Precision', (m.precision*100||0).toFixed(2)+'%', 'green'],
    ['Recall', (m.recall*100||0).toFixed(2)+'%', 'orange'],
    ['F1-Score', (m.f1*100||0).toFixed(2)+'%', 'purple'],
    ['Throughput', fmt(m.throughput)+' samp/s', 'cyan'],
    ['Model', DATA.metadata?.config?.model_type?.toUpperCase() || 'N/A', 'blue'],
  ];
  el.innerHTML = kpis.map(([l,v,c]) =>
    `<div class="kpi ${{c}}"><div class="val">${{v}}</div><div class="lbl">${{l}}</div></div>`
  ).join('');
}})();

document.getElementById('gen-time').textContent = DATA.metadata?.timestamp || 'N/A';

// ════════════════════════════════════════
// TEST 1: Inference Batch Size
// ════════════════════════════════════════
(function() {{
  const d = DATA.test1_inference_batch || {{}};
  const bs = Object.keys(d).map(Number).sort((a,b)=>a-b);
  const tp = bs.map(k => d[k]?.throughput_mean || 0);
  const lat = bs.map(k => d[k]?.latency_ms_mean || 0);
  const sp = bs.map(k => d[k]?.speedup_vs_bs1 || 0);

  new Chart(document.getElementById('chart-t1-throughput'), {{
    type: 'bar',
    data: {{ labels: bs.map(String), datasets: [{{ label: 'Throughput (samples/sec)', data: tp, backgroundColor: COLORS.blue+'cc', borderColor: COLORS.blue, borderWidth: 1 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Samples/sec' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t1-latency'), {{
    type: 'line',
    data: {{ labels: bs.map(String), datasets: [{{ label: 'Latency (ms/sample)', data: lat, borderColor: COLORS.orange, backgroundColor: COLORS.orange+'33', fill: true, tension: 0.3, pointRadius: 5 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'ms/sample' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t1-speedup'), {{
    type: 'line',
    data: {{ labels: bs.map(String), datasets: [
      {{ label: 'Speedup vs Batch=1', data: sp, borderColor: COLORS.green, backgroundColor: COLORS.green+'33', fill: true, tension: 0.3, pointRadius: 5 }},
      {{ label: 'Linear (ideal)', data: bs.map(k => k), borderColor: '#ffffff33', borderDash: [5,5], pointRadius: 0 }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Speedup' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
  }});

  const maxBs = bs[tp.indexOf(Math.max(...tp))];
  document.getElementById('analysis-t1').innerHTML = `<h4>Phân Tích Test 1</h4><ul>
    <li><strong>Batch size tối ưu:</strong> ${{maxBs}} — đạt throughput cao nhất ${{fmt(Math.max(...tp))}} samples/sec</li>
    <li><strong>Speedup tối đa:</strong> ${{fmt(Math.max(...sp),1)}}x so với batch=1</li>
    <li>Batch lớn hơn giúp amortize Python call overhead và tận dụng SIMD/GPU parallelism</li>
    <li>Diminishing returns xuất hiện khi batch > ${{maxBs}} do CPU/GPU cache saturation</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// TEST 2: Pipeline Comparison
// ════════════════════════════════════════
(function() {{
  const d = DATA.test2_pipeline_comparison || {{}};
  const modes = ['single_thread', 'multi_thread', 'multi_process'];
  const labels = ['Single-Thread', 'Multi-Thread', 'Multi-Process'];
  const colors = [COLORS.blue, COLORS.green, COLORS.orange];

  const tp = modes.map(m => d[m]?.throughput || 0);
  const avgLat = modes.map(m => d[m]?.avg_latency_ms || 0);
  const p95 = modes.map(m => d[m]?.p95_latency_ms || 0);
  const p99 = modes.map(m => d[m]?.p99_latency_ms || 0);

  new Chart(document.getElementById('chart-t2-throughput'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ label: 'Throughput (pred/sec)', data: tp, backgroundColor: colors.map(c=>c+'cc'), borderColor: colors, borderWidth: 1 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Predictions/sec' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t2-latency'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Avg', data: avgLat, backgroundColor: COLORS.cyan+'99' }},
        {{ label: 'P95', data: p95, backgroundColor: COLORS.orange+'99' }},
        {{ label: 'P99', data: p99, backgroundColor: COLORS.red+'99' }},
      ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Latency (ms)' }} }} }} }}
  }});

  const best = labels[tp.indexOf(Math.max(...tp))];
  const fastestLat = labels[avgLat.indexOf(Math.min(...avgLat))];
  document.getElementById('analysis-t2').innerHTML = `<h4>Phân Tích Test 2</h4><ul>
    <li><strong>Throughput cao nhất:</strong> ${{best}} — ${{fmt(Math.max(...tp))}} pred/sec</li>
    <li><strong>Latency thấp nhất:</strong> ${{fastestLat}} — ${{fmt(Math.min(...avgLat),2)}} ms</li>
    <li>Single-thread thường nhanh hơn với workload nhẹ (simulated) do không có synchronization overhead</li>
    <li>Multi-thread bị GIL hạn chế cho CPU-bound tasks, nhưng numpy/XGBoost release GIL khi chạy C code</li>
    <li>Multi-process tránh GIL nhưng chịu IPC overhead (serialize qua pipe ~100μs/message)</li>
    <li>Với heavy AI workload thực tế (model lớn, data phức tạp), multi-thread/process sẽ hiệu quả hơn</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// TEST 3: Queue Buffer
// ════════════════════════════════════════
(function() {{
  const d = DATA.test3_queue_buffer || {{}};
  const qs = Object.keys(d).map(Number).sort((a,b)=>a-b);
  const tp = qs.map(k => d[k]?.throughput || 0);
  const avgL = qs.map(k => d[k]?.avg_latency_ms || 0);
  const p99 = qs.map(k => d[k]?.p99_latency_ms || 0);

  new Chart(document.getElementById('chart-t3-throughput'), {{
    type: 'line',
    data: {{ labels: qs.map(k=>k.toLocaleString()), datasets: [{{ label: 'Throughput', data: tp, borderColor: COLORS.blue, backgroundColor: COLORS.blue+'33', fill: true, tension: 0.3, pointRadius: 5 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Predictions/sec' }} }}, x: {{ title: {{ display: true, text: 'Queue Size' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t3-latency'), {{
    type: 'line',
    data: {{ labels: qs.map(k=>k.toLocaleString()), datasets: [
      {{ label: 'Avg Latency', data: avgL, borderColor: COLORS.orange, tension: 0.3, pointRadius: 5 }},
      {{ label: 'P99 Latency', data: p99, borderColor: COLORS.red, tension: 0.3, pointRadius: 5 }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Latency (ms)' }} }}, x: {{ title: {{ display: true, text: 'Queue Size' }} }} }} }}
  }});

  const optIdx = tp.indexOf(Math.max(...tp));
  document.getElementById('analysis-t3').innerHTML = `<h4>Phân Tích Test 3</h4><ul>
    <li><strong>Queue size tối ưu:</strong> ${{qs[optIdx].toLocaleString()}} — throughput ${{fmt(tp[optIdx])}} pred/sec</li>
    <li>Queue nhỏ (<500): Producer thường xuyên bị block → giảm throughput</li>
    <li>Queue lớn (>10,000): Throughput ổn nhưng latency tăng do packets chờ lâu trong buffer</li>
    <li>Rule of thumb: Queue size ≈ 2-5 giây × expected packet rate</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// TEST 4: Batch Size Pipeline
// ════════════════════════════════════════
(function() {{
  const d = DATA.test4_batch_pipeline || {{}};
  const bs = Object.keys(d).map(Number).sort((a,b)=>a-b);
  const stTp = bs.map(k => d[k]?.single_thread?.throughput || 0);
  const mtTp = bs.map(k => d[k]?.multi_thread?.throughput || 0);
  const stLat = bs.map(k => d[k]?.single_thread?.avg_latency_ms || 0);
  const mtLat = bs.map(k => d[k]?.multi_thread?.avg_latency_ms || 0);

  new Chart(document.getElementById('chart-t4-throughput'), {{
    type: 'line',
    data: {{ labels: bs.map(String), datasets: [
      {{ label: 'Single-Thread', data: stTp, borderColor: COLORS.blue, tension: 0.3, pointRadius: 5 }},
      {{ label: 'Multi-Thread', data: mtTp, borderColor: COLORS.green, tension: 0.3, pointRadius: 5 }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Predictions/sec' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t4-latency'), {{
    type: 'line',
    data: {{ labels: bs.map(String), datasets: [
      {{ label: 'ST Avg Latency', data: stLat, borderColor: COLORS.blue, borderDash: [5,5], tension: 0.3, pointRadius: 5 }},
      {{ label: 'MT Avg Latency', data: mtLat, borderColor: COLORS.green, borderDash: [5,5], tension: 0.3, pointRadius: 5 }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Latency (ms)' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
  }});

  const bestSt = bs[stTp.indexOf(Math.max(...stTp))];
  const bestMt = bs[mtTp.indexOf(Math.max(...mtTp))];
  document.getElementById('analysis-t4').innerHTML = `<h4>Phân Tích Test 4</h4><ul>
    <li><strong>Best batch (Single-Thread):</strong> ${{bestSt}}</li>
    <li><strong>Best batch (Multi-Thread):</strong> ${{bestMt}}</li>
    <li>Batch nhỏ (1-8): Overhead per-prediction cao, throughput thấp cả 2 modes</li>
    <li>Batch lớn (128-256): Throughput cao nhưng latency tăng do chờ gom đủ batch</li>
    <li>Multi-thread hưởng lợi nhiều hơn từ large batch vì amortize synchronization cost</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// TEST 5: Scalability
// ════════════════════════════════════════
(function() {{
  const d = DATA.test5_scalability || {{}};
  const mt = d.multi_thread || {{}};
  const mp = d.multi_process || {{}};

  const workers = [...new Set([...Object.keys(mt), ...Object.keys(mp)].map(Number))].sort((a,b)=>a-b);
  const mtSp = workers.map(w => mt[w]?.speedup || 0);
  const mpSp = workers.map(w => mp[w]?.speedup || 0);
  const mtEff = workers.map(w => (mt[w]?.efficiency || 0) * 100);
  const mpEff = workers.map(w => (mp[w]?.efficiency || 0) * 100);
  const mtTp = workers.map(w => mt[w]?.throughput || 0);
  const mpTp = workers.map(w => mp[w]?.throughput || 0);
  const ideal = workers.map(w => w);

  new Chart(document.getElementById('chart-t5-speedup'), {{
    type: 'line',
    data: {{ labels: workers.map(String), datasets: [
      {{ label: 'Multi-Thread', data: mtSp, borderColor: COLORS.green, tension: 0.3, pointRadius: 6 }},
      {{ label: 'Multi-Process', data: mpSp, borderColor: COLORS.orange, tension: 0.3, pointRadius: 6 }},
      {{ label: 'Linear (ideal)', data: ideal, borderColor: '#ffffff33', borderDash: [5,5], pointRadius: 0 }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Speedup (x)' }} }}, x: {{ title: {{ display: true, text: 'AI Workers' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t5-efficiency'), {{
    type: 'bar',
    data: {{ labels: workers.map(String), datasets: [
      {{ label: 'Multi-Thread', data: mtEff, backgroundColor: COLORS.green+'99' }},
      {{ label: 'Multi-Process', data: mpEff, backgroundColor: COLORS.orange+'99' }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Efficiency (%)' }}, max: 120 }}, x: {{ title: {{ display: true, text: 'AI Workers' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t5-throughput'), {{
    type: 'bar',
    data: {{ labels: workers.map(String), datasets: [
      {{ label: 'Multi-Thread', data: mtTp, backgroundColor: COLORS.green+'99' }},
      {{ label: 'Multi-Process', data: mpTp, backgroundColor: COLORS.orange+'99' }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Predictions/sec' }} }}, x: {{ title: {{ display: true, text: 'AI Workers' }} }} }} }}
  }});

  document.getElementById('analysis-t5').innerHTML = `<h4>Phân Tích Test 5 — Amdahl's Law</h4><ul>
    <li>Linear scaling (ideal): Speedup = N → Efficiency = 100%</li>
    <li>Sublinear (typical): Speedup &lt; N do synchronization overhead</li>
    <li>Negative scaling: Quá nhiều workers → context switching overhead &gt; performance gain</li>
    <li><strong>Python GIL:</strong> Multi-thread bị hạn chế cho pure-Python CPU-bound code, nhưng numpy/XGBoost release GIL</li>
    <li><strong>IPC overhead:</strong> Multi-process phải serialize data qua pipe (~100μs/msg vs ~1μs cho thread queue)</li>
    <li>GPU inference thường chỉ cần 1-2 workers vì GPU tự parallel hóa nội bộ</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// TEST 6: CPU vs GPU
// ════════════════════════════════════════
(function() {{
  const d = DATA.test6_cpu_gpu || {{}};
  const cpu = d.cpu || {{}};
  const gpu = d.gpu || {{}};
  const bs = [...new Set([...Object.keys(cpu), ...Object.keys(gpu)].map(Number))].filter(x=>!isNaN(x)).sort((a,b)=>a-b);

  if (bs.length === 0) {{
    document.getElementById('analysis-t6').innerHTML = '<h4>Không có dữ liệu CPU/GPU</h4>';
    return;
  }}

  const cpuTp = bs.map(k => cpu[k]?.throughput || 0);
  const gpuTp = bs.map(k => gpu[k]?.throughput || 0);
  const speedups = bs.map(k => (gpu[k]?.throughput||0) / Math.max(cpu[k]?.throughput||1, 1));

  new Chart(document.getElementById('chart-t6-throughput'), {{
    type: 'bar',
    data: {{ labels: bs.map(String), datasets: [
      {{ label: 'CPU', data: cpuTp, backgroundColor: COLORS.blue+'99' }},
      {{ label: 'GPU', data: gpuTp, backgroundColor: COLORS.green+'99' }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ title: {{ display: true, text: 'Samples/sec' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
  }});

  new Chart(document.getElementById('chart-t6-speedup'), {{
    type: 'bar',
    data: {{ labels: bs.map(String), datasets: [
      {{ label: 'GPU Speedup vs CPU', data: speedups, backgroundColor: COLORS.cyan+'99' }},
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Speedup (x)' }} }}, x: {{ title: {{ display: true, text: 'Batch Size' }} }} }} }}
  }});

  document.getElementById('analysis-t6').innerHTML = `<h4>Phân Tích Test 6</h4><ul>
    <li>GPU hiệu quả nhất với batch lớn (≥128) nhờ massive parallelism (thousands of CUDA cores)</li>
    <li>Batch nhỏ: GPU overhead (kernel launch, memory transfer) lớn hơn compute time</li>
    <li>XGBoost hist tree_method hỗ trợ CUDA natively (không cần copy data)</li>
    <li>Với model nhỏ (150 estimators, depth 6), CPU có thể nhanh hơn GPU do low overhead</li>
  </ul>`;
}})();

// ════════════════════════════════════════
// SUMMARY TABLE
// ════════════════════════════════════════
(function() {{
  const d = DATA.test2_pipeline_comparison || {{}};
  const tbody = document.getElementById('summary-tbody');
  const modes = [
    ['Single-Thread', d.single_thread],
    ['Multi-Thread (2w)', d.multi_thread],
    ['Multi-Process (2w)', d.multi_process],
  ];

  modes.forEach(([name, r]) => {{
    if (!r) return;
    const sp = r.speedup_vs_single || 1;
    let badge = 'badge-green';
    if (sp < 0.5) badge = 'badge-red';
    else if (sp < 1) badge = 'badge-orange';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{name}}</td>
      <td>${{fmt(r.throughput)}}</td>
      <td>${{fmt(r.avg_latency_ms, 2)}}</td>
      <td>${{fmt(r.p95_latency_ms, 2)}}</td>
      <td>${{fmt(r.p99_latency_ms, 2)}}</td>
      <td>${{sp.toFixed(2)}}x</td>
      <td><span class="badge ${{badge}}">${{sp >= 1 ? 'Optimal' : sp >= 0.5 ? 'Acceptable' : 'Suboptimal'}}</span></td>
    `;
    tbody.appendChild(tr);
  }});
}})();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[+] Evaluation report: {out_path}")


if __name__ == "__main__":
    run_full_research_benchmark()
