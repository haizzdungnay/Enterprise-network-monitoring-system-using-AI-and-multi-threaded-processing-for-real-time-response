# 🛡️ Hệ Thống Giám Sát Mạng Doanh Nghiệp - AI & Đa Luồng

## Mục Lục
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Lý Thuyết Nền Tảng](#2-lý-thuyết-nền-tảng)
3. [Cài Đặt & Chạy](#3-cài-đặt--chạy)
4. [Chi Tiết Từng Module](#4-chi-tiết-từng-module)
5. [Benchmark & Đánh Giá](#5-benchmark--đánh-giá)

---

## 1. Tổng Quan Kiến Trúc

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐
│  NIC / PCAP │───▶│   Producer   │───▶│  Shared Queue   │───▶│ Consumer │
│  (Capture)  │    │  (Threading) │    │ (Thread-safe)   │    │ (Extract)│
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
2. Packet được đẩy vào Shared Queue (thread-safe)
3. Consumer lấy packet từ queue, trích xuất features
4. Features được batch lại, đẩy vào Feature Queue
5. AI Worker pool lấy batch features, chạy inference song song
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
- **Dataset CICIDS2017:** 80+ features trích xuất từ network flows
- **Dataset UNSW-NB15:** 49 features, 9 loại tấn công
- **Model:** Random Forest, XGBoost, hoặc Neural Network (MLP)
- **Features quan trọng:** Duration, Protocol, Bytes, Packets, Flags, ...

### 2.5. Tối Ưu Hóa
- **Queue Buffer Size:** Quá nhỏ → block producer, quá lớn → tốn RAM
- **Batch Size:** Nhóm nhiều sample để inference 1 lần → giảm overhead
- **Worker Pool Size:** Quá nhiều → context switching overhead

---

## 3. Cài Đặt & Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Bước 1: Train model trên dataset
python train_model.py

# Bước 2: Chạy benchmark so sánh single/multi thread/process
python benchmark.py

# Bước 3: Chạy hệ thống monitoring (cần dataset PCAP hoặc CSV)
python main_monitor.py

# Bước 4: Mở dashboard xem kết quả
# Mở file dashboard.html trong trình duyệt
```

---

## 4. Chi Tiết Từng Module

Xem comments trong từng file source code để hiểu lý thuyết kết hợp thực hành.

| File | Mô tả |
|------|--------|
| `config.py` | Cấu hình toàn hệ thống |
| `dataset_loader.py` | Load & preprocess CICIDS2017/UNSW-NB15 |
| `feature_extractor.py` | Trích xuất features từ packet/flow |
| `ai_model.py` | Train, save, load, inference AI model |
| `train_model.py` | Script train model trên dataset |
| `producer.py` | Packet capture producer |
| `consumer.py` | Feature extraction consumer |
| `ai_worker.py` | AI inference worker pool |
| `main_monitor.py` | Main orchestrator |
| `benchmark.py` | So sánh performance |

---

## 5. Benchmark & Đánh Giá

### Metrics đo lường:
- **Throughput:** Số packet/flow xử lý được mỗi giây
- **Latency:** Thời gian từ khi capture đến khi có kết quả AI
- **Scalability:** Throughput thay đổi thế nào khi tăng worker
- **CPU Usage:** Mức sử dụng CPU ở mỗi cấu hình
- **Memory Usage:** RAM tiêu thụ

### So sánh 3 mode:
1. **Single-thread:** Baseline, tuần tự capture → extract → inference
2. **Multi-thread:** Producer-Consumer với threading
3. **Multi-process:** Producer-Consumer với multiprocessing
