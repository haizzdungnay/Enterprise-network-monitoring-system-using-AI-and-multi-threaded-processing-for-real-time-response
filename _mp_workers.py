"""
Lightweight MP worker module for research benchmark.
Avoids importing torch/config/ai_model to keep child process startup fast.
"""
import os
import time
import numpy as np
import joblib


def mp_worker_simple(features_file, start_idx, end_idx, batch_size,
                     model_path, scaler_path, result_q, worker_id):
    """Simple MP worker: load model, process a slice of pre-generated features.
    
    Sends summary stats dict (small) instead of raw latency list to avoid
    pipe buffer deadlock on Windows.
    """
    try:
        # Load pre-generated features from shared numpy file
        all_features = np.load(features_file, mmap_mode='r')
        my_features = np.array(all_features[start_idx:end_idx])  # copy slice to RAM
        
        # Load model (CPU-only for child processes)
        model = joblib.load(model_path)
        if hasattr(model, 'set_params'):
            try:
                model.set_params(device='cpu', tree_method='hist')
            except Exception:
                pass
        scaler = joblib.load(scaler_path)
        exp = getattr(scaler, 'n_features_in_', 18)
        
        # Align features
        if my_features.shape[1] != exp:
            if my_features.shape[1] < exp:
                my_features = np.pad(my_features,
                                     ((0, 0), (0, exp - my_features.shape[1])),
                                     mode='constant')
            else:
                my_features = my_features[:, :exp]
        
        # Process in batches, track latency
        total_pred = 0
        latencies = []
        
        for i in range(0, len(my_features), batch_size):
            batch = my_features[i:i + batch_size]
            t0 = time.perf_counter()
            X_scaled = scaler.transform(batch)
            model.predict(X_scaled)
            t1 = time.perf_counter()
            lat = t1 - t0
            total_pred += len(batch)
            latencies.append(lat)
        
        lat_arr = np.array(latencies)
        # Per-sample latencies
        n = total_pred
        total_lat = float(lat_arr.sum())
        per_sample_lats = []
        for lat, bs_actual in zip(latencies,
            [min(batch_size, len(my_features) - i) 
             for i in range(0, len(my_features), batch_size)]):
            per_sample_lats.extend([lat / bs_actual] * bs_actual)
        ps_arr = np.array(per_sample_lats)
        
        result_q.put({
            'predictions': total_pred,
            'total_latency_sec': total_lat,
            'avg_latency_ms': float(ps_arr.mean() * 1000) if len(ps_arr) > 0 else 0.0,
            'p50_latency_ms': float(np.percentile(ps_arr, 50) * 1000) if len(ps_arr) > 0 else 0.0,
            'p95_latency_ms': float(np.percentile(ps_arr, 95) * 1000) if len(ps_arr) > 0 else 0.0,
            'p99_latency_ms': float(np.percentile(ps_arr, 99) * 1000) if len(ps_arr) > 0 else 0.0,
            'min_latency_ms': float(ps_arr.min() * 1000) if len(ps_arr) > 0 else 0.0,
            'max_latency_ms': float(ps_arr.max() * 1000) if len(ps_arr) > 0 else 0.0,
            'worker_id': worker_id,
        })
    except Exception as e:
        result_q.put({
            'predictions': 0,
            'total_latency_sec': 0.0,
            'avg_latency_ms': 0.0, 'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0, 'p99_latency_ms': 0.0,
            'min_latency_ms': 0.0, 'max_latency_ms': 0.0,
            'worker_id': worker_id,
            'error': str(e),
        })
