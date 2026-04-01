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
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else "None"
except ImportError:
    _CUDA_AVAILABLE = False
    _GPU_NAME = "None (torch not installed)"

# ═══════════════════════════════════════════════════════════════
# 1. ĐƯỜNG DẪN DỮ LIỆU
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
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
BATCH_SIZE = 256 if _CUDA_AVAILABLE else 64   # GPU: 256, CPU: 64
BATCH_TIMEOUT = 0.1 if _CUDA_AVAILABLE else 0.5  # GPU xử lý nhanh hơn

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
USE_GPU = _CUDA_AVAILABLE          # Auto-detect CUDA (RTX 3070 Ti)
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

# XGBoost params – GPU 3070 Ti: tree_method='hist' + device='cuda'
XGB_N_ESTIMATORS = 300            # Tăng lên vì GPU train nhanh hơn
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.05          # Nhỏ hơn → chính xác hơn (đủ estimators)
XGB_TREE_METHOD = 'hist'          # 'hist' hỗ trợ cả CPU lẫn CUDA
XGB_DEVICE = GPU_DEVICE           # 'cuda' hoặc 'cpu' theo auto-detect

# MLP params (sklearn, CPU only)
MLP_HIDDEN_LAYERS = (128, 64, 32)
MLP_MAX_ITER = 300

# PyTorch MLP params (GPU-accelerated)
TORCH_HIDDEN_LAYERS = (256, 128, 64)  # Lớn hơn vì GPU có đủ tài nguyên
TORCH_EPOCHS = 50
TORCH_LR = 1e-3
TORCH_WEIGHT_DECAY = 1e-4
TORCH_BATCH_TRAIN = 2048 if _CUDA_AVAILABLE else 512  # Mini-batch size khi train
TORCH_AMP = _CUDA_AVAILABLE       # Automatic Mixed Precision (float16 trên 3070 Ti)

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
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
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
    if USE_GPU and MODEL_TYPE == 'torch_mlp':
        print(f"  Torch AMP (fp16):    {TORCH_AMP}")
        print(f"  Torch Train Batch:   {TORCH_BATCH_TRAIN}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
