"""
╔══════════════════════════════════════════════════════════════════╗
║           CONFIG.PY - CẤU HÌNH HỆ THỐNG GIÁM SÁT              ║
╚══════════════════════════════════════════════════════════════════╝

LÝ THUYẾT: Tại sao cần file config riêng?
─────────────────────────────────────────
Trong hệ thống thực tế, các tham số như queue size, batch size, số worker
cần được điều chỉnh (tuning) tùy theo:
  - Phần cứng (CPU cores, RAM, có GPU không)
  - Tải mạng (bao nhiêu packet/giây)
  - Yêu cầu latency (cần phản hồi trong bao lâu)

Việc tách config ra file riêng giúp:
  1. Dễ thay đổi mà không sửa code
  2. Dễ tạo các profile khác nhau (dev, staging, production)
  3. Dễ tự động hóa tuning
"""

import os
import multiprocessing

# ── Auto-detect GPU (PyTorch / XGBoost CUDA) ──
try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else "None"
    _GPU_VRAM_MB = (torch.cuda.get_device_properties(0).total_memory // 1024**2
                    if _CUDA_AVAILABLE else 0)
except Exception:
    _CUDA_AVAILABLE = False
    _GPU_NAME = "None (torch not installed)"
    _GPU_VRAM_MB = 0

# ═══════════════════════════════════════════════════════════════
# GPU PROFILES - Tự động chọn tham số phù hợp theo loại GPU
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: GPU Profiles
───────────────────────
Mỗi GPU có lượng VRAM và compute capability khác nhau.
Việc dùng cùng batch size / hidden layers cho mọi GPU sẽ gây:
  - OOM (Out of Memory) trên GPU yếu (RTX 3050 4GB)
  - Lãng phí tài nguyên trên GPU mạnh (RTX 4090 24GB)

Phân loại theo VRAM:
  - LOW    (≤ 4GB):  RTX 3050 Laptop, GTX 1650, MX550...
  - MEDIUM (≤ 6GB):  RTX 3060 Laptop 6GB, GTX 1660...
  - HIGH   (≤ 8GB):  RTX 3070 Ti, RTX 3070, RTX 4060...
  - ULTRA  (> 8GB):  RTX 3090, RTX 4080, RTX 4090...

Người dùng cũng có thể override bằng biến môi trường:
  GPU_PROFILE=low python train_model.py
"""

GPU_PROFILES = {
    "low": {        # ≤ 4GB VRAM (RTX 3050 Laptop 4GB, GTX 1650, ...)
        "batch_size": 128,
        "batch_timeout": 0.2,
        "torch_hidden_layers": (128, 64, 32),
        "torch_batch_train": 512,
        "torch_amp": True,          # AMP giúp tiết kiệm VRAM đáng kể
        "torch_epochs": 30,
        "xgb_n_estimators": 150,
        "xgb_max_depth": 6,
        "description": "Low VRAM (≤4GB): RTX 3050 Laptop, GTX 1650, MX550",
    },
    "medium": {     # ≤ 6GB VRAM (RTX 3060 Laptop 6GB, GTX 1660, ...)
        "batch_size": 192,
        "batch_timeout": 0.15,
        "torch_hidden_layers": (192, 96, 48),
        "torch_batch_train": 1024,
        "torch_amp": True,
        "torch_epochs": 40,
        "xgb_n_estimators": 200,
        "xgb_max_depth": 7,
        "description": "Medium VRAM (≤6GB): RTX 3060 Laptop, GTX 1660",
    },
    "high": {       # ≤ 8GB VRAM (RTX 3070 Ti, RTX 3070, RTX 4060, ...)
        "batch_size": 256,
        "batch_timeout": 0.1,
        "torch_hidden_layers": (256, 128, 64),
        "torch_batch_train": 2048,
        "torch_amp": True,
        "torch_epochs": 50,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 8,
        "description": "High VRAM (≤8GB): RTX 3070 Ti, RTX 3070, RTX 4060",
    },
    "ultra": {      # > 8GB VRAM (RTX 3090, RTX 4080, RTX 4090, ...)
        "batch_size": 512,
        "batch_timeout": 0.05,
        "torch_hidden_layers": (512, 256, 128),
        "torch_batch_train": 4096,
        "torch_amp": True,
        "torch_epochs": 60,
        "xgb_n_estimators": 500,
        "xgb_max_depth": 10,
        "description": "Ultra VRAM (>8GB): RTX 3090, RTX 4080, RTX 4090",
    },
}


def _detect_gpu_profile():
    """
    Tự động chọn GPU profile dựa trên VRAM.
    Có thể override bằng biến môi trường GPU_PROFILE.
    """
    # Cho phép user override
    env_profile = os.environ.get("GPU_PROFILE", "").strip().lower()
    if env_profile in GPU_PROFILES:
        return env_profile

    if not _CUDA_AVAILABLE:
        return "cpu"

    if _GPU_VRAM_MB <= 4096:
        return "low"
    elif _GPU_VRAM_MB <= 6144:
        return "medium"
    elif _GPU_VRAM_MB <= 8192:
        return "high"
    else:
        return "ultra"


_GPU_PROFILE_NAME = _detect_gpu_profile()
_GPU_PROFILE = GPU_PROFILES.get(_GPU_PROFILE_NAME, None)

# ═══════════════════════════════════════════════════════════════
# 1. ĐƯỜNG DẪN DỮ LIỆU
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "MachineLearningCSV", "MachineLearningCVE")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Tạo thư mục nếu chưa có
for d in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 2. CẤU HÌNH PACKET CAPTURE
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: Network Interface
────────────────────────────
- 'eth0': Interface Ethernet mặc định trên Linux
- 'wlan0': Interface WiFi
- 'lo': Loopback (localhost)
- None: Scapy sẽ tự chọn interface mặc định

SNAP_LEN: Số byte tối đa capture từ mỗi packet
- 65535: Capture toàn bộ packet
- 128: Chỉ capture header (tiết kiệm RAM, đủ cho phân tích flow)
"""
NETWORK_INTERFACE = None          # None = auto-detect
SNAP_LEN = 65535                  # Max bytes per packet
CAPTURE_TIMEOUT = 60              # Giây - thời gian capture mỗi session
PCAP_FILE = None                  # Đường dẫn file PCAP (None = live capture)

# ═══════════════════════════════════════════════════════════════
# 3. CẤU HÌNH QUEUE & BUFFER
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: Queue Sizing
───────────────────────
Queue là bộ đệm giữa Producer và Consumer:

  Quá nhỏ (100):
    → Producer bị block khi queue đầy
    → Có thể mất packet
    → Latency thấp (packet được xử lý ngay)

  Quá lớn (100000):
    → Producer không bao giờ bị block
    → Tốn nhiều RAM
    → Latency cao (packet nằm chờ lâu trong queue)

  Tối ưu: Đủ để hấp thụ burst traffic nhưng không quá lớn
  Rule of thumb: 2-5 giây × expected_packet_rate

BATCH SIZE:
  Nhóm nhiều sample lại → 1 lần inference
  - Batch nhỏ (1-8): Latency thấp, throughput thấp
  - Batch lớn (64-256): Latency cao hơn, throughput cao
  - Optimal: Phụ thuộc model và hardware
"""
PACKET_QUEUE_SIZE = 10000         # Packet queue capacity
FEATURE_QUEUE_SIZE = 5000         # Feature queue capacity
BATCH_SIZE = _GPU_PROFILE["batch_size"] if _GPU_PROFILE else 64
BATCH_TIMEOUT = _GPU_PROFILE["batch_timeout"] if _GPU_PROFILE else 0.5

# ═══════════════════════════════════════════════════════════════
# 4. CẤU HÌNH WORKER POOL
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: Worker Pool Sizing
─────────────────────────────
Số worker tối ưu phụ thuộc vào:

  I/O-bound tasks (capture, ghi file):
    → workers = 2 × CPU_cores (vì thread chờ I/O nhiều)

  CPU-bound tasks (feature extraction, AI inference):
    → workers = CPU_cores (tối đa, tránh context switching)

  GPU tasks:
    → Thường 1-2 worker gửi batch lên GPU là đủ
    → GPU tự song song hóa bên trong

Context Switching Overhead:
  - Mỗi lần OS chuyển từ thread A sang thread B mất ~1-10μs
  - Nếu có quá nhiều thread → spend more time switching than working
"""
NUM_CPU_CORES = multiprocessing.cpu_count()
NUM_CAPTURE_THREADS = 1            # Thường 1 là đủ cho 1 NIC
NUM_CONSUMER_THREADS = 2           # Feature extraction workers
# GPU tự song song hóa nội bộ → chỉ cần 1 worker gửi batch lên GPU
NUM_AI_WORKERS = 1 if _CUDA_AVAILABLE else max(1, NUM_CPU_CORES - 3)
USE_GPU = _CUDA_AVAILABLE
GPU_DEVICE = 'cuda' if _CUDA_AVAILABLE else 'cpu'

# ═══════════════════════════════════════════════════════════════
# 5. CẤU HÌNH AI MODEL
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: Model Selection
──────────────────────────
Cho Network Intrusion Detection:

  Random Forest:
    + Nhanh inference, dễ train
    + Xử lý tốt tabular data
    - Tốn RAM với nhiều cây

  XGBoost:
    + Accuracy cao nhất cho tabular data
    + Inference nhanh
    - Train chậm hơn RF

  Neural Network (MLP):
    + Có thể tận dụng GPU
    + Scale tốt với data lớn
    - Cần feature engineering cẩn thận
    - Dễ overfit nếu data ít
"""
MODEL_TYPE = "xgboost"             # "random_forest", "xgboost", "mlp", "torch_mlp"
MODEL_PATH = os.path.join(MODEL_DIR, "ids_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

# Random Forest params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_N_JOBS = -1                    # -1 = dùng tất cả CPU cores

# XGBoost params – tự động điều chỉnh theo GPU profile
XGB_N_ESTIMATORS = _GPU_PROFILE["xgb_n_estimators"] if _GPU_PROFILE else 100
XGB_MAX_DEPTH = _GPU_PROFILE["xgb_max_depth"] if _GPU_PROFILE else 6
XGB_LEARNING_RATE = 0.05          # Nhỏ hơn → chính xác hơn (đủ estimators)
XGB_TREE_METHOD = 'hist'          # 'hist' hỗ trợ cả CPU lẫn CUDA
XGB_DEVICE = GPU_DEVICE           # 'cuda' hoặc 'cpu' theo auto-detect

# MLP params (sklearn, CPU only)
MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_MAX_ITER = 300

# PyTorch MLP params – tự động điều chỉnh theo GPU profile
TORCH_HIDDEN_LAYERS = _GPU_PROFILE["torch_hidden_layers"] if _GPU_PROFILE else (128, 64, 32)
TORCH_EPOCHS = _GPU_PROFILE["torch_epochs"] if _GPU_PROFILE else 30
TORCH_LR = 1e-3
TORCH_WEIGHT_DECAY = 1e-4
TORCH_BATCH_TRAIN = _GPU_PROFILE["torch_batch_train"] if _GPU_PROFILE else 512
TORCH_AMP = _GPU_PROFILE["torch_amp"] if _GPU_PROFILE else False

# ═══════════════════════════════════════════════════════════════
# 6. CẤU HÌNH DATASET
# ═══════════════════════════════════════════════════════════════
"""
LÝ THUYẾT: Dataset cho IDS
──────────────────────────
CICIDS2017:
  - 80+ features từ CICFlowMeter
  - ~2.8M flows
  - 14 loại tấn công + Benign
  - Realistic enterprise traffic

UNSW-NB15:
  - 49 features
  - ~2.5M records
  - 9 loại tấn công + Normal
  - Modern attack patterns

Cả 2 dataset đều sử dụng FLOW-BASED features:
  Flow = tập hợp các packet cùng (src_ip, dst_ip, src_port, dst_port, protocol)
  Features được tính từ flow: duration, byte count, packet count, flags, ...
"""
DATASET_TYPE = "cicids2017"        # "cicids2017" hoặc "unsw_nb15"
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")  # Merged cache (tự động tạo)
TEST_SIZE = 0.2                    # 20% cho test
RANDOM_STATE = 42

# Label mapping
LABEL_NORMAL = 0
LABEL_ATTACK = 1

# ═══════════════════════════════════════════════════════════════
# 7. CẤU HÌNH BENCHMARK
# ═══════════════════════════════════════════════════════════════
BENCHMARK_NUM_PACKETS = 50000      # Số packet giả lập cho benchmark
BENCHMARK_REPEAT = 3               # Số lần lặp để tính trung bình
BENCHMARK_WARMUP = 1000            # Packet warmup trước khi đo

# ═══════════════════════════════════════════════════════════════
# 8. CẤU HÌNH LOGGING & ALERT
# ═══════════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOG_DIR, "monitor.log")
ALERT_THRESHOLD = 0.7             # Probability threshold cho attack alert
STATS_INTERVAL = 5                # Giây - in stats mỗi 5 giây


def print_config():
    """In ra toàn bộ cấu hình hiện tại."""
    print("=" * 60)
    print("       CẤU HÌNH HỆ THỐNG GIÁM SÁT MẠNG")
    print("=" * 60)
    print(f"  CPU Cores:           {NUM_CPU_CORES}")
    print(f"  Capture Threads:     {NUM_CAPTURE_THREADS}")
    print(f"  Consumer Threads:    {NUM_CONSUMER_THREADS}")
    print(f"  AI Workers:          {NUM_AI_WORKERS}")
    print(f"  Use GPU:             {USE_GPU}  [{_GPU_NAME}]")
    print(f"  GPU VRAM:            {_GPU_VRAM_MB} MB")
    print(f"  GPU Profile:         {_GPU_PROFILE_NAME.upper()}")
    if _GPU_PROFILE:
        print(f"                       {_GPU_PROFILE['description']}")
    print(f"  GPU Device:          {GPU_DEVICE}")
    print(f"  Packet Queue Size:   {PACKET_QUEUE_SIZE}")
    print(f"  Feature Queue Size:  {FEATURE_QUEUE_SIZE}")
    print(f"  Batch Size:          {BATCH_SIZE}")
    print(f"  Batch Timeout:       {BATCH_TIMEOUT}s")
    print(f"  Model Type:          {MODEL_TYPE}")
    print(f"  Dataset:             {DATASET_TYPE}")
    if USE_GPU and MODEL_TYPE == 'xgboost':
        print(f"  XGB Device:          {XGB_DEVICE}")
        print(f"  XGB Tree Method:     {XGB_TREE_METHOD}")
        print(f"  XGB Estimators:      {XGB_N_ESTIMATORS}")
        print(f"  XGB Max Depth:       {XGB_MAX_DEPTH}")
    if USE_GPU and MODEL_TYPE == 'torch_mlp':
        print(f"  Torch AMP (fp16):    {TORCH_AMP}")
        print(f"  Torch Train Batch:   {TORCH_BATCH_TRAIN}")
        print(f"  Torch Hidden Layers: {TORCH_HIDDEN_LAYERS}")
    print(f"  Override: set GPU_PROFILE=low|medium|high|ultra")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
