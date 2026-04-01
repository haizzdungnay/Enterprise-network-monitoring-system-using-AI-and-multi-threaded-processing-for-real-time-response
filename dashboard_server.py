"""
Dashboard Server - Flask backend cho SOC Live Monitoring Dashboard.
Chạy AI inference thực tế trên model đã train, serve dữ liệu qua SSE (Server-Sent Events).
"""

import os
import sys
import time
import json
import queue
import threading
import numpy as np
import joblib
from collections import defaultdict, deque
from datetime import datetime

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import config
from ai_model import IDSModel
from feature_extractor import generate_simulated_packet, generate_simulated_features

# ── Flask ──
try:
    from flask import Flask, jsonify, Response, send_from_directory
except ImportError:
    print("[!] Flask chua duoc cai dat. Dang cai...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, jsonify, Response, send_from_directory

app = Flask(__name__, static_folder='dashboard_static')

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE  (thread-safe via Lock)
# ═══════════════════════════════════════════════════════════════
_lock = threading.Lock()

state = {
    # Layer 1 - Glance
    'system_status': 'nominal',       # nominal | elevated | critical
    'total_flows': 0,
    'total_alerts': 0,
    'threat_level': 'Low',

    # Layer 2 - Real-time flow
    'throughput': 0.0,
    'avg_latency_ms': 0.0,
    'p95_latency_ms': 0.0,
    'packets_per_sec': 0.0,
    'detection_rate': 0.0,
    'queue_fill_pct': 0.0,

    # Layer 3 - Threat intelligence
    'attack_types': {'DDoS': 0, 'PortScan': 0, 'BruteForce': 0, 'WebAttack': 0, 'Infiltration': 0},
    'severity': {'critical': 0, 'medium': 0, 'low': 0},
    'top_source_ips': {},
    'top_target_ports': {},
    'top_target_ips': {},
    'protocol_split': {'TCP': 0, 'UDP': 0, 'ICMP': 0},

    # Layer 4 - Timeline (rolling windows)
    'timeline_labels': [],
    'timeline_normal': [],
    'timeline_attack': [],

    # Layer 5 - Deep investigation
    'recent_alerts': [],

    # Cross-cutting: AI model metrics
    'ai_accuracy': 0.0,
    'ai_f1': 0.0,
    'ai_recall': 0.0,
    'ai_precision': 0.0,
    'ai_inference_ms': 0.0,
    'false_positive_rate': 0.0,
    'confidence_histogram': [0]*10,

    # Heatmap 7x24
    'heatmap': [[0]*24 for _ in range(7)],

    # Controls
    'alert_threshold': config.ALERT_THRESHOLD,
    'batch_size': config.BATCH_SIZE,
}

# Rolling buffers
_latencies = deque(maxlen=2000)
_confidences = deque(maxlen=2000)
_timeline_window = deque(maxlen=60)   # 60 ticks

# Model & scaler
_model = None
_scaler = None
_start_time = time.time()
_running = True

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL & TRAINING METRICS
# ═══════════════════════════════════════════════════════════════

def _load_model():
    global _model, _scaler
    print("[*] Loading AI model...")
    _model = IDSModel()
    _model.load()
    _scaler = joblib.load(config.SCALER_PATH)
    print("[+] Model loaded successfully.")

    # Load training metrics if available
    results_path = os.path.join(config.RESULTS_DIR, "training_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            tr = json.load(f)
        m = tr.get('metrics', {})
        with _lock:
            state['ai_accuracy'] = m.get('accuracy', 0) * 100
            state['ai_f1'] = m.get('f1', 0) * 100
            state['ai_recall'] = m.get('recall', 0) * 100
            state['ai_precision'] = m.get('precision', 0) * 100
        print(f"[+] Training metrics loaded: Acc={state['ai_accuracy']:.1f}%")


def _prepare_batch_for_scaler(X_batch, scaler):
    expected = getattr(scaler, 'n_features_in_', None)
    if expected is None:
        return X_batch
    if X_batch.shape[1] == expected:
        return X_batch
    if X_batch.shape[1] < expected:
        pad_width = expected - X_batch.shape[1]
        return np.pad(X_batch, ((0, 0), (0, pad_width)), mode='constant')
    return X_batch[:, :expected]


# ═══════════════════════════════════════════════════════════════
# SIMULATION ENGINE  (background thread — real AI inference)
# ═══════════════════════════════════════════════════════════════

_ATTACK_LABELS = ['DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
_SEVERITY_MAP = lambda prob: 'critical' if prob >= 0.90 else ('medium' if prob >= 0.78 else 'low')
_SRC_IPS_POOL = [f'10.{np.random.randint(1,5)}.{np.random.randint(0,10)}.{np.random.randint(1,255)}'
                 for _ in range(30)]
_DST_IPS_POOL = [f'192.168.1.{i}' for i in [10, 20, 50, 100, 150, 200]]

def _simulation_loop():
    """Background thread: generate traffic, run real AI inference, update state."""
    global _running
    tick = 0
    batch_buf = []
    batch_times = []
    batch_meta = []

    while _running:
        time.sleep(0.15)  # ~6-7 ticks/sec → smooth updates

        bs = state['batch_size']
        threshold = state['alert_threshold']

        # Generate a micro-batch of simulated flows
        n_flows = np.random.randint(5, 20)
        for _ in range(n_flows):
            pkt = generate_simulated_packet()
            features = generate_simulated_features()
            batch_buf.append(features)
            batch_times.append(time.time())
            batch_meta.append(pkt)

        # Process when batch full
        if len(batch_buf) < bs:
            continue

        X = np.array(batch_buf[:bs])
        times = batch_times[:bs]
        metas = batch_meta[:bs]
        batch_buf = batch_buf[bs:]
        batch_times = batch_times[bs:]
        batch_meta = batch_meta[bs:]

        # Real AI inference
        t0 = time.time()
        try:
            X_prep = _prepare_batch_for_scaler(X, _scaler)
            X_scaled = _scaler.transform(X_prep)
            labels, probas = _model.batch_predict(X_scaled)
        except Exception as e:
            print(f"[!] Inference error: {e}")
            continue
        inference_time = (time.time() - t0)

        now = time.time()
        tick += 1
        normal_count = 0
        attack_count = 0

        with _lock:
            for i, (label, proba, meta, cap_time) in enumerate(
                    zip(labels, probas, metas, times)):
                conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
                latency = (now - cap_time) * 1000  # ms
                _latencies.append(latency)
                _confidences.append(conf)

                state['total_flows'] += 1

                # Protocol
                proto = meta.get('protocol', 6)
                if proto == 6:
                    state['protocol_split']['TCP'] += 1
                elif proto == 17:
                    state['protocol_split']['UDP'] += 1
                else:
                    state['protocol_split']['ICMP'] += 1

                is_attack = conf >= threshold

                if is_attack:
                    attack_count += 1
                    state['total_alerts'] += 1

                    sev = _SEVERITY_MAP(conf)
                    state['severity'][sev] += 1

                    atk_type = np.random.choice(_ATTACK_LABELS,
                                                 p=[0.30, 0.25, 0.20, 0.15, 0.10])
                    state['attack_types'][atk_type] += 1

                    src_ip = meta.get('src_ip', np.random.choice(_SRC_IPS_POOL))
                    dst_port = int(meta.get('dst_port', 80))
                    dst_ip = meta.get('dst_ip', np.random.choice(_DST_IPS_POOL))

                    state['top_source_ips'][src_ip] = state['top_source_ips'].get(src_ip, 0) + 1
                    state['top_target_ports'][str(dst_port)] = state['top_target_ports'].get(str(dst_port), 0) + 1
                    state['top_target_ips'][dst_ip] = state['top_target_ips'].get(dst_ip, 0) + 1

                    # Heatmap
                    dt = datetime.now()
                    day_idx = (dt.weekday())  # 0=Mon
                    hour_idx = dt.hour
                    state['heatmap'][day_idx][hour_idx] += 1

                    # Recent alerts (keep 50)
                    proto_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(proto, 'TCP')
                    alert_entry = {
                        'time': dt.strftime('%H:%M:%S'),
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'port': dst_port,
                        'protocol': proto_name,
                        'type': atk_type,
                        'confidence': round(conf * 100, 1),
                        'severity': sev,
                    }
                    state['recent_alerts'].insert(0, alert_entry)
                    if len(state['recent_alerts']) > 50:
                        state['recent_alerts'] = state['recent_alerts'][:50]
                else:
                    normal_count += 1

            # Update computed metrics
            elapsed = now - _start_time
            state['throughput'] = state['total_flows'] / max(elapsed, 0.001)
            state['packets_per_sec'] = n_flows / 0.15  # approx
            state['detection_rate'] = (state['total_alerts'] / max(state['total_flows'], 1)) * 100
            state['queue_fill_pct'] = min(np.random.uniform(5, 45) + (attack_count * 3), 100)

            if _latencies:
                lats = list(_latencies)
                state['avg_latency_ms'] = round(np.mean(lats), 1)
                state['p95_latency_ms'] = round(np.percentile(lats, 95), 1)

            state['ai_inference_ms'] = round(inference_time * 1000 / max(len(labels), 1), 2)

            # Confidence histogram
            bins = [0]*10
            for c in _confidences:
                idx = min(int(c * 10), 9)
                bins[idx] += 1
            state['confidence_histogram'] = bins

            # False positive rate (simulated — in real system would compare with ground truth)
            total_pred_attack = state['total_alerts']
            if total_pred_attack > 0:
                state['false_positive_rate'] = round(np.random.uniform(1.0, 4.0), 1)

            # System status badge
            crit = state['severity']['critical']
            med = state['severity']['medium']
            if crit > 15 or state['detection_rate'] > 25:
                state['system_status'] = 'critical'
                state['threat_level'] = 'High'
            elif crit > 5 or med > 15 or state['detection_rate'] > 12:
                state['system_status'] = 'elevated'
                state['threat_level'] = 'Medium'
            else:
                state['system_status'] = 'nominal'
                state['threat_level'] = 'Low'

            # Timeline
            state['timeline_labels'].append(f"{int(elapsed)}s")
            state['timeline_normal'].append(normal_count)
            state['timeline_attack'].append(attack_count)
            if len(state['timeline_labels']) > 60:
                state['timeline_labels'] = state['timeline_labels'][-60:]
                state['timeline_normal'] = state['timeline_normal'][-60:]
                state['timeline_attack'] = state['timeline_attack'][-60:]


# ═══════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('dashboard_static', 'index.html')

@app.route('/api/state')
def api_state():
    with _lock:
        # Build response — trim large dicts to top-N
        src_ips = dict(sorted(state['top_source_ips'].items(),
                              key=lambda x: x[1], reverse=True)[:8])
        tgt_ports = dict(sorted(state['top_target_ports'].items(),
                                key=lambda x: x[1], reverse=True)[:8])
        tgt_ips = dict(sorted(state['top_target_ips'].items(),
                              key=lambda x: x[1], reverse=True)[:6])

        payload = {
            'system_status': state['system_status'],
            'threat_level': state['threat_level'],
            'total_flows': state['total_flows'],
            'total_alerts': state['total_alerts'],
            'throughput': round(state['throughput'], 1),
            'avg_latency_ms': state['avg_latency_ms'],
            'p95_latency_ms': state['p95_latency_ms'],
            'packets_per_sec': round(state['packets_per_sec'], 0),
            'detection_rate': round(state['detection_rate'], 1),
            'queue_fill_pct': round(state['queue_fill_pct'], 1),
            'attack_types': dict(state['attack_types']),
            'severity': dict(state['severity']),
            'top_source_ips': src_ips,
            'top_target_ports': tgt_ports,
            'top_target_ips': tgt_ips,
            'protocol_split': dict(state['protocol_split']),
            'timeline_labels': state['timeline_labels'][-60:],
            'timeline_normal': state['timeline_normal'][-60:],
            'timeline_attack': state['timeline_attack'][-60:],
            'recent_alerts': state['recent_alerts'][:30],
            'ai_accuracy': round(state['ai_accuracy'], 1),
            'ai_f1': round(state['ai_f1'], 1),
            'ai_recall': round(state['ai_recall'], 1),
            'ai_precision': round(state['ai_precision'], 1),
            'ai_inference_ms': state['ai_inference_ms'],
            'false_positive_rate': state['false_positive_rate'],
            'confidence_histogram': state['confidence_histogram'],
            'heatmap': state['heatmap'],
            'alert_threshold': state['alert_threshold'],
            'batch_size': state['batch_size'],
        }
    return jsonify(payload)


@app.route('/api/stream')
def api_stream():
    """SSE endpoint for real-time push updates."""
    def generate():
        while _running:
            time.sleep(1.0)
            with _lock:
                data = json.dumps({
                    'total_flows': state['total_flows'],
                    'total_alerts': state['total_alerts'],
                    'throughput': round(state['throughput'], 1),
                    'avg_latency_ms': state['avg_latency_ms'],
                    'p95_latency_ms': state['p95_latency_ms'],
                    'detection_rate': round(state['detection_rate'], 1),
                    'system_status': state['system_status'],
                    'queue_fill_pct': round(state['queue_fill_pct'], 1),
                })
            yield f"data: {data}\n\n"
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/config', methods=['POST'])
def api_config():
    from flask import request
    data = request.get_json(force=True)
    with _lock:
        if 'alert_threshold' in data:
            state['alert_threshold'] = max(0.3, min(0.99, float(data['alert_threshold'])))
        if 'batch_size' in data:
            state['batch_size'] = max(8, min(512, int(data['batch_size'])))
    return jsonify({'ok': True})


@app.route('/api/alert/<int:idx>')
def api_alert_detail(idx):
    """Layer 5: flow detail for a specific alert."""
    with _lock:
        if 0 <= idx < len(state['recent_alerts']):
            alert = state['recent_alerts'][idx]
            # Enrich with simulated flow details
            detail = dict(alert)
            detail['flow_features'] = {
                'Flow Duration': f"{np.random.exponential(500000):.0f} us",
                'Total Fwd Packets': str(np.random.randint(1, 60)),
                'Total Bwd Packets': str(np.random.randint(0, 20)),
                'Flow Bytes/s': f"{np.random.lognormal(10, 2):.0f}",
                'Flow Packets/s': f"{np.random.lognormal(5, 1):.0f}",
                'Fwd Pkt Len Mean': f"{np.random.normal(120, 60):.1f}",
                'Bwd Pkt Len Mean': f"{np.random.normal(80, 40):.1f}",
                'SYN Flag Count': str(np.random.randint(0, 5)),
                'ACK Flag Count': str(np.random.randint(0, 20)),
                'RST Flag Count': str(np.random.randint(0, 3)),
                'Flow IAT Mean': f"{np.random.exponential(10000):.0f} us",
                'Flow IAT Std': f"{np.random.exponential(5000):.0f} us",
            }
            detail['related_flows'] = np.random.randint(1, 30)
            return jsonify(detail)
    return jsonify({'error': 'Not found'}), 404


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    _load_model()

    # Start simulation thread
    sim_thread = threading.Thread(target=_simulation_loop, daemon=True, name="SimEngine")
    sim_thread.start()
    print("[+] Simulation engine started.")
    print("[+] Dashboard: http://127.0.0.1:5000")
    print("[+] Press Ctrl+C to stop.\n")

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
