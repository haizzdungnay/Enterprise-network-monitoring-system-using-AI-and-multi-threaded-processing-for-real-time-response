# 🛡️ Hệ Thống Giám Sát Mạng Doanh Nghiệp - AI & Đa Luồng

## Mục Lục
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Lý Thuyết Nền Tảng](#2-lý-thuyết-nền-tảng)
3. [Cài Đặt & Chạy](#3-cài-đặt--chạy)
4. [Cấu Trúc Dự Án](#4-cấu-trúc-dự-án)
5. [Dataset CICIDS2017](#5-dataset-cicids2017)
6. [GPU & Cấu Hình](#6-gpu--cấu-hình)
7. [Benchmark & Đánh Giá](#7-benchmark--đánh-giá)
8. [Live SOC Dashboard](#8-live-soc-dashboard)

---

## 1. Tổng Quan Kiến Trúc

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐
│  NIC / PCAP │───▶│   Producer   │───▶│  Packet Queue   │───▶│ Consumer │
│  (Capture)  │    │  (Threading) │    │  (Thread-safe)  │    │ (Extract)│
└─────────────┘    └──────────────┘    └─────────────────┘    └────┬─────┘
                                                                    │
                                               ┌────────────────────┘
                                               ▼
                                     ┌──────────────────┐    ┌────────────┐
                                     │  Feature Queue   │───▶│ AI Workers │
                                     │  (Batch Buffer)  │    │   (Pool)   │
                                     └──────────────────┘    └─────┬──────┘
                                                                   │
                                               ┌───────────────────┘
                                               ▼
                                     ┌──────────────────┐
                                     │   Alert System   │
                                     │  (Dashboard/Log) │
                                     └──────────────────┘
```

**Luồng dữ liệu:**
1. Producer bắt packet từ NIC (hoặc đọc từ PCAP file)
2. Packet được đẩy vào Packet Queue (thread-safe)
3. Consumer lấy packet từ queue, trích xuất vector feature (18 features canonical)
4. Features được gom thành batch, đẩy vào Feature Queue
5. AI Worker pool lấy batch features, chạy inference song song (CPU hoặc GPU)
6. Kết quả phân loại được gửi tới Alert System

---

## 2. Lý Thuyết Nền Tảng

### 2.1. Packet Capture
- **NIC (Network Interface Card):** Card mạng cho phép bắt gói tin
- **Promiscuous Mode:** Chế độ cho phép NIC bắt TẤT CẢ packet trên segment mạng
- **PCAP Format:** Định dạng chuẩn để lưu trữ packet đã bắt được
- **Scapy / PyShark:** Thư viện Python để capture & parse packet

### 2.2. Producer-Consumer Pattern
- **Producer:** Thread/Process tạo ra dữ liệu (capture packet)
- **Consumer:** Thread/Process xử lý dữ liệu (extract features)
- **Queue:** Buffer trung gian, đảm bảo thread-safety
- **Tại sao cần pattern này?** Vì tốc độ capture ≠ tốc độ xử lý.
  Nếu xử lý chậm hơn capture → cần buffer để không mất packet.

### 2.3. Threading vs Multiprocessing
| Tiêu chí | Threading | Multiprocessing |
|-----------|-----------|-----------------|
| GIL | Bị ảnh hưởng | Không bị |
| Shared Memory | Dễ dàng | Cần IPC |
| I/O-bound | Hiệu quả | Overhead cao |
| CPU-bound | Kém (do GIL) | Hiệu quả |
| Overhead | Thấp | Cao |

**GIL (Global Interpreter Lock):** Python chỉ cho phép 1 thread chạy
Python bytecode tại một thời điểm. → Threading không tăng tốc CPU-bound
tasks, nhưng hiệu quả cho I/O-bound (network capture, disk I/O).

### 2.4. AI Model cho Network Intrusion Detection
- **Dataset CICIDS2017:** 79 features raw từ CICFlowMeter, nhưng pipeline train/inference dùng 18 features canonical nhất quán
- **Dataset UNSW-NB15:** 49 features, 9 loại tấn công (hỗ trợ thêm)
- **Model:** Random Forest, XGBoost, MLP (sklearn), hoặc PyTorch MLP
- **Features quan trọng:** Flow Duration, Protocol, Bytes, Packets, Flags, IAT, ...

### 2.5. Tối Ưu Hóa
- **Queue Buffer Size:** Quá nhỏ → block producer, quá lớn → tốn RAM
- **Batch Size:** Nhóm nhiều sample để inference 1 lần → giảm overhead
- **Worker Pool Size:** Quá nhiều → context switching overhead
- **GPU Batching:** Gửi batch lớn lên GPU để tận dụng tối đa CUDA cores

---

## 3. Cài Đặt & Chạy

### 3.1. Yêu cầu
- Python ≥ 3.9
- GPU NVIDIA (tùy chọn, tự động detect)

### 3.2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**PyTorch (GPU)** — chọn đúng phiên bản CUDA:

```bash
# CUDA 12.x (driver ≥ 525 — RTX 30/40 series)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (driver cũ hơn — GTX 10/16, RTX 20/30)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Kiểm tra sau khi cài:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.3. Dataset

Dataset **CICIDS2017** đã được đặt đúng vị trí tại:

```
MachineLearningCSV/
└── MachineLearningCVE/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Hệ thống tự động load và merge toàn bộ 8 file khi train. Lần đầu chạy sẽ tạo
file cache tại `data/dataset.csv` để các lần sau load nhanh hơn.

### 3.4. Các bước chạy

```bash
# Bước 1: Train model (tự động load dataset từ MachineLearningCSV/)
python train_model.py

# Bước 2: Chạy benchmark so sánh single/multi thread/process
python benchmark.py

# Bước 3: Chạy hệ thống monitoring thời gian thực
python main_monitor.py
```

---

## 4. Cấu Trúc Dự Án

```
.
├── MachineLearningCSV/
│   └── MachineLearningCVE/        # Dataset CICIDS2017 (8 CSV files)
├── data/                           # Cache & processed data (tự động tạo)
├── models/                         # Model, scaler, feature names (tự động tạo)
├── logs/                           # Log files (tự động tạo)
├── results/                        # Benchmark results (tự động tạo)
│
├── config.py                       # Cấu hình toàn hệ thống
├── dataset_loader.py               # Load & preprocess CICIDS2017/UNSW-NB15
├── feature_extractor.py            # Trích xuất features từ packet/flow
├── ai_model.py                     # Train, save, load, inference AI model
├── train_model.py                  # Script train model trên dataset
├── main_monitor.py                 # Main orchestrator (Producer-Consumer-AI)
├── benchmark.py                    # So sánh performance các mode
└── requirements.txt
```

| File | Mô tả |
|------|--------|
| `config.py` | Cấu hình hệ thống, GPU profile tự động, đường dẫn dataset |
| `dataset_loader.py` | Load 8 CSV CICIDS2017, merge, preprocess, train/test split |
| `feature_extractor.py` | Trích xuất 79 features CICFlowMeter từ packet/flow thực tế |
| `ai_model.py` | RandomForest / XGBoost / MLP / PyTorch MLP |
| `train_model.py` | Pipeline train đầy đủ với GPU warmup và evaluation |
| `main_monitor.py` | Orchestrator: Producer → Queue → Consumer → AI → Alert |
| `benchmark.py` | Benchmark throughput single/multi thread/process |
| `dashboard_server.py` | Flask backend phục vụ dashboard live + API realtime |
| `dashboard_static/index.html` | Giao diện SOC dashboard 5 layers chạy trên localhost |

---

## 5. Dataset CICIDS2017

### Cấu trúc
Dataset do **Canadian Institute for Cybersecurity** tạo ra, ghi lại traffic mạng
thực tế trong 5 ngày làm việc:

| File | Nội dung |
|------|----------|
| Monday | Traffic bình thường (BENIGN) |
| Tuesday | Brute Force — FTP-Patator, SSH-Patator |
| Wednesday | DoS/DDoS — Slowloris, Slowhttptest, Hulk, GoldenEye |
| Thursday | Web Attacks (XSS, SQLi, Brute Force) + Infiltration |
| Friday (sáng) | Traffic bình thường |
| Friday (chiều) | PortScan, DDoS (LOIT) |

### Đặc điểm
- **~2.8 triệu flows** tổng cộng từ 8 file
- **79 features raw** trích xuất bởi CICFlowMeter (flow-level, không packet-level)
- **18 features canonical** được chọn để đồng bộ giữa training và realtime inference
- **14 loại tấn công** + BENIGN
- Label: `BENIGN` = 0, tất cả tấn công = 1 (binary classification)

### Pipeline load dữ liệu

```
dataset_loader.load_dataset()
    │
    ├── Nếu có data/dataset.csv  →  load cache (nhanh)
    │
    ├── Nếu có MachineLearningCSV/MachineLearningCVE/  →  merge 8 CSV
    │       └── Lưu cache vào data/dataset.csv
    │
    └── Nếu không có gì  →  tạo synthetic data (100K flows, chỉ để demo)
```

### Preprocessing
1. Strip whitespace khỏi tên cột
2. Loại bỏ columns không phải feature (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`)
3. Xử lý `Inf`, `-Inf` → NaN, sau đó drop rows có NaN
4. Stratified train/test split (80/20)
5. `StandardScaler` fit trên train, transform cả train+test (tránh data leakage)
6. Lưu scaler và feature names vào `models/` để dùng khi inference

---

## 6. GPU & Cấu Hình

### GPU Profiles (tự động detect theo VRAM)

| Profile | VRAM | GPU tiêu biểu | Batch Size | XGB Estimators |
|---------|------|---------------|------------|----------------|
| `low` | ≤ 4 GB | RTX 3050 Laptop, GTX 1650 | 128 | 150 |
| `medium` | ≤ 6 GB | RTX 3060 Laptop, GTX 1660 | 192 | 200 |
| `high` | ≤ 8 GB | RTX 3070 Ti, RTX 4060 | 256 | 300 |
| `ultra` | > 8 GB | RTX 3090, RTX 4080/4090 | 512 | 500 |

Override thủ công:
```bash
GPU_PROFILE=low python train_model.py
```

### Các tham số quan trọng trong `config.py`

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `DATASET_DIR` | `MachineLearningCSV/MachineLearningCVE` | Thư mục chứa 8 CSV CICIDS2017 |
| `DATASET_PATH` | `data/dataset.csv` | Cache merged dataset |
| `MODEL_TYPE` | `xgboost` | `random_forest` / `xgboost` / `mlp` / `torch_mlp` |
| `DATASET_TYPE` | `cicids2017` | `cicids2017` hoặc `unsw_nb15` |
| `PACKET_QUEUE_SIZE` | 10000 | Dung lượng packet queue |
| `FEATURE_QUEUE_SIZE` | 5000 | Dung lượng feature queue |
| `ALERT_THRESHOLD` | 0.7 | Ngưỡng xác suất để cảnh báo tấn công |

---

## 7. Benchmark & Đánh Giá

### Metrics đo lường
- **Throughput:** Số packet/flow xử lý được mỗi giây
- **Latency:** Thời gian từ khi capture đến khi có kết quả AI
- **Accuracy / Precision / Recall / F1:** Chất lượng phân loại
- **Confusion Matrix:** TP / TN / FP / FN
- **CPU / GPU Usage:** Mức tận dụng phần cứng

### So sánh 3 mode
1. **Single-thread:** Baseline, tuần tự capture → extract → inference
2. **Multi-thread:** Producer-Consumer với threading (I/O-bound friendly)
3. **Multi-process:** Producer-Consumer với multiprocessing (CPU-bound, bypass GIL)

### Chạy benchmark
```bash
python benchmark.py
```
Kết quả lưu tại `results/`.

---

## 8. Live SOC Dashboard

Dashboard live đã được tích hợp theo mô hình 5 tầng metrics (Glance → Real-time Flow → Threat Intelligence → Timeline/Patterns → Deep Investigation).

### Cách chạy

```bash
python dashboard_server.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

### Tính năng chính
- Layer 1: system status badge, total alerts, throughput, latency, detection rate
- Layer 2: timeline realtime, attack types, confidence distribution
- Layer 3: severity/protocol split, top source IPs, top target ports/internal IPs, recent alerts
- Layer 4: weekly heatmap, AI model metrics, điều chỉnh alert threshold + batch size
- Layer 5: click alert để xem chi tiết flow và confidence phục vụ điều tra
