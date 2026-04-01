"""
╔══════════════════════════════════════════════════════════════════╗
║        AI_MODEL.PY - TRAIN & INFERENCE CHO IDS                 ║
╚══════════════════════════════════════════════════════════════════╝

LÝ THUYẾT: Machine Learning cho Intrusion Detection
────────────────────────────────────────────────────

Bài toán IDS là BINARY CLASSIFICATION:
  Input:  Vector features (19 chiều trong project này)
  Output: 0 (Normal) hoặc 1 (Attack)

Tại sao chọn từng model?
─────────────────────────

RANDOM FOREST:
  - Tập hợp nhiều Decision Trees
  - Mỗi cây train trên random subset của data + features
  - Kết quả = majority vote của tất cả cây
  - Ưu điểm: Robust, ít overfit, nhanh inference
  - Nhược: Tốn RAM khi nhiều cây

XGBOOST (Extreme Gradient Boosting):
  - Cũng là tập hợp cây, nhưng train TUẦN TỰ
  - Mỗi cây mới sửa lỗi của cây trước
  - Ưu điểm: SOTA cho tabular data, accuracy cao
  - Nhược: Train chậm hơn, cần tuning hyper-params

MLP (Multi-Layer Perceptron):
  - Neural network cơ bản (fully connected)
  - Layers: Input → Hidden1 → Hidden2 → Hidden3 → Output
  - Activation: ReLU (hidden), Sigmoid (output)
  - Ưu điểm: Flexible, GPU-acceleratable
  - Nhược: Cần nhiều data, dễ overfit

INFERENCE OPTIMIZATION:
───────────────────────
- BATCHING: Nhóm nhiều samples → 1 lần predict → amortize overhead
- MODEL CACHING: Load model 1 lần, dùng nhiều lần
- NUMPY VECTORIZATION: sklearn predict() đã vectorized internally
"""

import time
import threading
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import config
import warnings
warnings.filterwarnings('ignore')

# ── Thử import XGBoost (optional) ──
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] XGBoost chua duoc cai dat. Su dung: pip install xgboost")

# ── Thử import PyTorch (optional, GPU support) ──
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    if torch.cuda.is_available():
        print(f"[+] GPU detected: {torch.cuda.get_device_name(0)} "
              f"| VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
except ImportError:
    HAS_TORCH = False
    print("[!] PyTorch chua duoc cai dat. GPU MLP se khong kha dung.")


class _TorchMLP(nn.Module if HAS_TORCH else object):
    """
    PyTorch MLP với GPU support.
    Tận dụng CUDA Tensor Cores của RTX 3070 Ti cho fast training & inference.
    """
    def __init__(self, input_dim, hidden_layers, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class IDSModel:
    """
    ╔════════════════════════════════════════════════════════╗
    ║  Intrusion Detection System Model                     ║
    ╠════════════════════════════════════════════════════════╣
    ║  Wrapper class cho các ML models.                     ║
    ║  Hỗ trợ: train, predict, batch predict, save/load    ║
    ╚════════════════════════════════════════════════════════╝
    """

    def __init__(self, model_type=None):
        self.model_type = model_type or config.MODEL_TYPE
        self.model = None
        self.scaler = None
        self.feature_names = None
        # GPU không thread-safe khi nhiều thread gọi đồng thời
        self._gpu_lock = threading.Lock() if config.USE_GPU else None
        self._create_model()

    def _create_model(self):
        """
        Khởi tạo model theo config.

        LÝ THUYẾT: Hyperparameter quan trọng
        ────────────────────────────────────
        Random Forest:
          - n_estimators: Số cây. Nhiều hơn = chính xác hơn nhưng chậm hơn
          - max_depth: Độ sâu tối đa. Sâu quá → overfit, nông quá → underfit
          - n_jobs=-1: Train song song trên tất cả CPU cores

        XGBoost:
          - learning_rate: Bước học. Nhỏ = chậm hội tụ nhưng chính xác
          - n_estimators: Số boosting rounds
          - max_depth: Tương tự RF

        MLP:
          - hidden_layer_sizes: Kiến trúc mạng (128, 64, 32)
            = 3 hidden layers với 128, 64, 32 neurons
          - max_iter: Số epoch tối đa
        """
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                n_jobs=config.RF_N_JOBS,
                random_state=config.RANDOM_STATE,
                class_weight='balanced'  # Xử lý class imbalance
            )
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=config.XGB_LEARNING_RATE,
                tree_method=config.XGB_TREE_METHOD,  # 'hist' hỗ trợ CUDA
                device=config.XGB_DEVICE,             # 'cuda' → GPU 3070 Ti
                random_state=config.RANDOM_STATE,
                scale_pos_weight=4,  # ~80% normal, 20% attack → ratio 4:1
                eval_metric='logloss',
                n_jobs=-1
            )
        elif self.model_type == "torch_mlp":
            # PyTorch MLP – GPU-accelerated, khởi tạo sau khi biết n_features
            self.model = None  # Sẽ tạo trong train() khi biết input_dim
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
                max_iter=config.MLP_MAX_ITER,
                random_state=config.RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        else:
            print(f"[!] Model type '{self.model_type}' không khả dụng. Dùng Random Forest.")
            self.model_type = "random_forest"
            self._create_model()

    def train(self, X_train, y_train):
        """
        Train model trên training data.

        LÝ THUYẾT: Training Process
        ───────────────────────────
        1. Model nhận X_train (features) và y_train (labels)
        2. Học patterns phân biệt Normal vs Attack
        3. Lưu learned parameters (tree structures, weights, ...)
        4. Sau train, model có thể predict trên data mới
        """
        print(f"\n[*] Training {self.model_type}...")
        print(f"    Training samples: {len(X_train)}")

        start_time = time.time()

        if self.model_type == "torch_mlp" and HAS_TORCH:
            train_time = self._train_torch(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
            train_time = time.time() - start_time

        print(f"    Training time: {train_time:.2f}s")

        # Feature importance (cho RF và XGBoost)
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            print(f"    Top 5 important features:")
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    print(f"      {i+1}. {self.feature_names[idx]}: "
                          f"{importances[idx]:.4f}")

        return train_time

    def _train_torch(self, X_train, y_train):
        """
        Train PyTorch MLP trên GPU.

        LÝ THUYẾT: GPU Training với AMP
        ────────────────────────────────
        - Automatic Mixed Precision (AMP): dùng float16 thay float32
          → 3070 Ti có Tensor Cores tối ưu cho float16
          → Gần 2x tốc độ, dùng nửa VRAM
        - DataLoader pin_memory=True: pre-load batch lên pinned RAM
          → Transfer CPU→GPU nhanh hơn qua DMA
        - num_workers=0 trên Windows (fork không được hỗ trợ tốt)
        """
        device = torch.device(config.GPU_DEVICE)
        input_dim = X_train.shape[1]

        self.model = _TorchMLP(input_dim, config.TORCH_HIDDEN_LAYERS).to(device)
        self._torch_device = device

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config.TORCH_BATCH_TRAIN,
                            shuffle=True, num_workers=0,
                            pin_memory=(config.GPU_DEVICE == 'cuda'))

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=config.TORCH_LR,
                                      weight_decay=config.TORCH_WEIGHT_DECAY)
        # Class imbalance: pos_weight = 80%/20% = 4
        pos_weight = torch.tensor([4.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=config.TORCH_AMP)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.TORCH_LR,
            steps_per_epoch=len(loader), epochs=config.TORCH_EPOCHS)

        start = time.time()
        self.model.train()
        for epoch in range(config.TORCH_EPOCHS):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=config.TORCH_AMP):
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"    Epoch {epoch+1:>3}/{config.TORCH_EPOCHS} | "
                      f"loss={avg_loss:.4f}")
        self.model.eval()
        return time.time() - start

    def evaluate(self, X_test, y_test):
        """
        Đánh giá model trên test data.

        LÝ THUYẾT: Metrics cho IDS
        ──────────────────────────
        - Accuracy: % dự đoán đúng tổng thể
          → Misleading khi class imbalance (99% normal → 99% acc chỉ cần predict normal)

        - Precision: Trong tất cả dự đoán Attack, bao nhiêu % đúng là Attack?
          → Quan trọng: Giảm FALSE POSITIVE (cảnh báo giả)

        - Recall (Sensitivity): Trong tất cả Attack thật, bao nhiêu % được phát hiện?
          → Quan trọng: Giảm FALSE NEGATIVE (bỏ sót tấn công)

        - F1-Score: Harmonic mean của Precision và Recall
          → Cân bằng giữa 2 metrics trên

        Trong IDS thực tế:
          - Recall quan trọng hơn Precision (bỏ sót attack nguy hiểm hơn cảnh báo giả)
          - Nhưng quá nhiều false positive → alert fatigue → bỏ qua cảnh báo thật
        """
        print(f"\n[*] Evaluating {self.model_type}...")

        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = time.time() - start_time

        # Tính metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)

        print(f"    ┌─────────────────────────────────┐")
        print(f"    │  Accuracy:   {accuracy:.4f}             │")
        print(f"    │  Precision:  {precision:.4f}             │")
        print(f"    │  Recall:     {recall:.4f}             │")
        print(f"    │  F1-Score:   {f1:.4f}             │")
        print(f"    └─────────────────────────────────┘")
        print(f"    Inference time: {inference_time:.4f}s for {len(X_test)} samples")
        print(f"    Throughput: {len(X_test)/inference_time:.0f} samples/sec")
        print(f"\n    Confusion Matrix:")
        print(f"    {'':>12} Predicted")
        print(f"    {'':>12} Normal  Attack")
        print(f"    Actual Normal  {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"    Actual Attack  {cm[1][0]:>6}  {cm[1][1]:>6}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'inference_time': inference_time,
            'throughput': len(X_test) / inference_time
        }

    def predict(self, X):
        """Predict labels cho input data."""
        if self._gpu_lock:
            with self._gpu_lock:
                if self.model_type == "torch_mlp" and HAS_TORCH:
                    return self._torch_predict_labels(X)
                return self.model.predict(X)
        if self.model_type == "torch_mlp" and HAS_TORCH:
            return self._torch_predict_labels(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probability (xác suất) cho từng class.

        LÝ THUYẾT: Probability vs Hard Label
        ────────────────────────────────────
        - predict():       [0, 1, 1, 0]  → Hard labels
        - predict_proba(): [[0.9, 0.1], [0.2, 0.8], ...]  → Soft probabilities

        Soft probabilities hữu ích hơn vì:
        1. Có thể điều chỉnh threshold: prob > 0.7 → attack (thay vì 0.5)
        2. Đánh giá confidence: prob=0.99 (rất chắc) vs prob=0.51 (không chắc)
        3. Alert prioritization: prob cao → cảnh báo ưu tiên
        """
        if self.model_type == "torch_mlp" and HAS_TORCH:
            return self._torch_predict_proba(X)
        return self.model.predict_proba(X)

    def _torch_predict_proba(self, X):
        """GPU inference: trả về [[p_normal, p_attack], ...] như sklearn."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        device = getattr(self, '_torch_device', torch.device('cpu'))
        with torch.no_grad():
            t = torch.tensor(X).to(device)
            with torch.cuda.amp.autocast(enabled=config.TORCH_AMP):
                logits = self.model(t)
            p_attack = torch.sigmoid(logits).cpu().numpy()
        p_normal = 1.0 - p_attack
        return np.stack([p_normal, p_attack], axis=1)

    def _torch_predict_labels(self, X):
        """GPU inference: trả về hard labels 0/1."""
        proba = self._torch_predict_proba(X)
        return (proba[:, 1] >= config.ALERT_THRESHOLD).astype(int)

    def batch_predict(self, X_batch):
        """
        Batch prediction - core function cho real-time monitoring.

        LÝ THUYẾT: Batch Inference
        ─────────────────────────
        GPU: torch.no_grad() + autocast(fp16) → Tensor Cores 3070 Ti
        CPU: numpy vectorization → SIMD/cache-friendly
        Speed up: 10-100x với batch_size=256
        """
        if len(X_batch) == 0:
            return np.array([]), np.array([])

        X_batch = np.array(X_batch)
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)

        labels = self.predict(X_batch)
        probas = self.predict_proba(X_batch)

        return labels, probas

    def evaluate(self, X_test, y_test):
        """
        Đánh giá model trên test data.

        LÝ THUYẾT: Metrics cho IDS
        ──────────────────────────
        - Precision: Giảm FALSE POSITIVE (cảnh báo giả)
        - Recall:    Giảm FALSE NEGATIVE (bỏ sót tấn công)
        - F1-Score:  Cân bằng giữa 2 metrics trên
        """
        print(f"\n[*] Evaluating {self.model_type}...")

        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"    ┌─────────────────────────────────┐")
        print(f"    │  Accuracy:   {accuracy:.4f}             │")
        print(f"    │  Precision:  {precision:.4f}             │")
        print(f"    │  Recall:     {recall:.4f}             │")
        print(f"    │  F1-Score:   {f1:.4f}             │")
        print(f"    └─────────────────────────────────┘")
        print(f"    Inference time: {inference_time:.4f}s for {len(X_test)} samples")
        print(f"    Throughput: {len(X_test)/inference_time:.0f} samples/sec")
        print(f"\n    Confusion Matrix:")
        print(f"    {'':>12} Predicted")
        print(f"    {'':>12} Normal  Attack")
        print(f"    Actual Normal  {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"    Actual Attack  {cm[1][0]:>6}  {cm[1][1]:>6}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'inference_time': inference_time,
            'throughput': len(X_test) / inference_time
        }

    def save(self, path=None):
        """Lưu model ra file."""
        path = path or config.MODEL_PATH
        if self.model_type == "torch_mlp" and HAS_TORCH:
            torch_path = path.replace('.joblib', '_torch.pt')
            torch.save(self.model.state_dict(), torch_path)
            print(f"[+] PyTorch model saved: {torch_path}")
        else:
            joblib.dump(self.model, path)
            print(f"[+] Model saved: {path}")

    def load(self, path=None):
        """Load model từ file."""
        path = path or config.MODEL_PATH
        if self.model_type == "torch_mlp" and HAS_TORCH:
            torch_path = path.replace('.joblib', '_torch.pt')
            input_dim = len(self.feature_names) if self.feature_names else 19
            device = torch.device(config.GPU_DEVICE)
            self.model = _TorchMLP(input_dim, config.TORCH_HIDDEN_LAYERS).to(device)
            self.model.load_state_dict(torch.load(torch_path, map_location=device))
            self.model.eval()
            self._torch_device = device
            print(f"[+] PyTorch model loaded: {torch_path}")
        else:
            self.model = joblib.load(path)
            # XGBoost: model có thể được train trên CUDA nhưng inference
            # từ numpy array cần chạy trên CPU để tránh device-mismatch crash
            if self.model_type == "xgboost" and HAS_XGBOOST:
                self.model.set_params(device='cpu')
                self._gpu_lock = None  # Inference trên CPU, không cần lock
            print(f"[+] Model loaded: {path}")

    def load_scaler(self, path=None):
        """Load scaler từ file."""
        path = path or config.SCALER_PATH
        self.scaler = joblib.load(path)
        self.feature_names = joblib.load(config.FEATURE_NAMES_PATH)
        print(f"[+] Scaler loaded: {path}")


def benchmark_single_vs_batch(model, X_test,
                              batch_sizes=None):
    """
    So sánh performance: single prediction vs batch prediction.

    LÝ THUYẾT: Tại sao batch nhanh hơn?
    ────────────────────────────────────
    CPU hoạt động hiệu quả nhất khi:
    1. Data nằm liên tục trong memory (cache-friendly)
    2. Thực hiện cùng 1 operation trên nhiều data (SIMD)
    3. Ít overhead function call

    GPU (RTX 3070 Ti): batch 256-512 tận dụng 5888 CUDA cores.
    """
    if batch_sizes is None:
        # GPU: test thêm batch lớn hơn để tận dụng VRAM 8GB
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512] if config.USE_GPU \
                      else [1, 8, 16, 32, 64, 128]

    print("\n" + "=" * 60)
    print("  BENCHMARK: Single vs Batch Prediction")
    print(f"  Device: {'GPU (' + config.GPU_DEVICE + ')' if config.USE_GPU else 'CPU'}")
    print("=" * 60)

    results = {}

    for bs in batch_sizes:
        n_batches = len(X_test) // bs
        if n_batches == 0:
            continue

        X_batched = X_test[:n_batches * bs]

        start = time.time()
        if bs == 1:
            # Single prediction
            for i in range(min(1000, len(X_batched))):
                model.predict(X_batched[i:i+1])
            elapsed = time.time() - start
            throughput = min(1000, len(X_batched)) / elapsed
        else:
            for i in range(0, min(1000*bs, len(X_batched)), bs):
                model.predict(X_batched[i:i+bs])
            elapsed = time.time() - start
            throughput = min(1000*bs, len(X_batched)) / elapsed

        results[bs] = {
            'throughput': throughput,
            'latency_per_sample': 1.0/throughput * 1000  # ms
        }

        print(f"  Batch size {bs:>4}: "
              f"Throughput = {throughput:>10,.0f} samples/sec | "
              f"Latency = {1.0/throughput*1000:.3f} ms/sample")

    return results
