# CHANGELOG — Hệ Thống Giám Sát Mạng Doanh Nghiệp

Ghi lại toàn bộ thay đổi theo thứ tự thời gian.
Format: `[FILE] Mô tả thay đổi`

---

## [2026-04-03] — Xác nhận train thật với CICIDS2017 + fix đường dẫn dataset

### Vấn đề phát hiện
- `DATASET_DIR` trong `config.py` trỏ đến `MachineLearningCSV/MachineLearningCVE/`
  nhưng thực tế folder đặt tại `MachineLearningCVE/` (không có thư mục cha `MachineLearningCSV`).
- `load_dataset()` fallback sang synthetic data mà **không báo lỗi** rõ ràng.
- Hàm `load_cicids2017_dir()` load toàn bộ 2.8M rows → tốn ~6-8GB RAM,
  không khả thi trên máy tính thông thường.

### Thay đổi

#### `config.py`
- **Sửa** `DATASET_DIR = os.path.join(BASE_DIR, "MachineLearningCVE")` — bỏ cấp cha `MachineLearningCSV/`
- **Thêm** `CICIDS_SAMPLE_PER_FILE = 100000` — giới hạn 100K rows/file khi merge
  (800K tổng, đủ đại diện, RAM ~1.5GB thay vì 6-8GB)

#### `dataset_loader.py`
- **Thêm** hàm `load_cicids2017_folder()` với stratified sampling:
  - Đọc từng file, sample theo đúng tỷ lệ class trong file đó
  - Hỗ trợ encoding utf-8 → latin-1 fallback
  - Tự phát hiện tên cột Label (có thể có leading space trong một số file)
- **Cập nhật** `load_dataset()`: ưu tiên `DATASET_DIR` trước `DATASET_PATH`
  khi cache chưa có, tự lưu cache sau khi merge

### Kết quả xác nhận trên RTX 3070 Ti

| Chỉ số | Giá trị |
|--------|---------|
| Dataset | 799,334 flows thật — 8 files CICIDS2017 |
| Features | 18 canonical (khớp với PacketFeatureExtractor) |
| Attack types | 14 loại: DDoS, PortScan, DoS Hulk, FTP-Patator, SSH-Patator, Bot, Web Attack... |
| Model | XGBoost CUDA (device='cuda', tree_method='hist') |
| Training time | **2.71 giây** trên GPU |
| Accuracy | **99.91%** |
| Precision | **99.58%** |
| Recall | **99.94%** — chỉ bỏ sót 17/30,779 tấn công |
| F1-Score | **99.76%** |
| Inference throughput | 780,000 samples/giây |

---

## [2026-04-03] — Research Benchmark & Evaluation Dashboard

### Thay đổi chính trong ngày

#### `research_benchmark.py` (mới)
- **Thêm** bộ đánh giá nghiên cứu toàn diện 6 bài test:
  - **Test 1**: Inference Batch Size Sweep (1→512) — throughput scaling ~247x
  - **Test 2**: Pipeline Comparison ST vs MT vs MP — so sánh 3 chế độ xử lý
  - **Test 3**: Queue Buffer Size Impact — tìm queue size tối ưu
  - **Test 4**: Batch Size Pipeline — trade-off throughput vs latency
  - **Test 5**: Scalability — đo hiệu quả khi tăng số workers (1/2/4)
  - **Test 6**: CPU vs GPU Inference — so sánh XGBoost trên CPU vs CUDA
- **Kết quả chính**: ST 16,092 pred/s, MT 14,559 pred/s, MP 5,722 pred/s
- **Tự động sinh** HTML report với Chart.js charts cho mọi bài test

#### `_mp_workers.py` (mới)
- **Thêm** module lightweight cho multi-process workers
- **Thiết kế** direct-split: pre-generate features → chia slice cho N workers
- **Tránh** pipe buffer deadlock trên Windows bằng summary dict (~200 bytes)
- **Không import** torch/config để giảm child process startup time

#### `dashboard_static/evaluation.html` (mới/cập nhật)
- **Sinh tự động** từ benchmark với 12+ biểu đồ Chart.js
- **Hiển thị** hardware info, model metrics, training results
- **Responsive** dark theme matching SOC dashboard

#### `dashboard_server.py` (cập nhật)
- **Thêm** route `/evaluation` — trang đánh giá nghiên cứu
- **Thêm** API `/api/benchmark` — chạy benchmark từ dashboard

#### `config.py` + `ai_model.py` (cập nhật)
- **Sửa** CUDA detection robustness: thêm `device_count() > 0` guard
- **Sửa** exception handling cho GPU initialization failures

#### `results/research_benchmark.json` (mới)
- **Lưu** toàn bộ kết quả benchmark dạng structured JSON

---

## [2026-04-02] — Hoàn thiện Live SOC Dashboard 5 tầng + chạy thực tế

### Thay đổi chính trong ngày

#### `dashboard_server.py` (mới)
- **Thêm** Flask backend cho dashboard realtime tại `http://127.0.0.1:5000`.
- **Thêm** engine mô phỏng traffic + chạy **AI inference thực tế** bằng model đã train (`ids_model.joblib` + `scaler.joblib`).
- **Thêm** API:
  - `/api/state`: trả toàn bộ dữ liệu dashboard theo cấu trúc 5 layers.
  - `/api/stream`: SSE stream realtime.
  - `/api/config`: cập nhật `alert_threshold` và `batch_size` từ UI.
  - `/api/alert/<idx>`: trả chi tiết alert cho deep investigation.

#### `dashboard_static/index.html` (mới)
- **Xây dựng** SOC dashboard đầy đủ 5 tầng metrics theo taxonomy trong DEMO.
- **Hiển thị realtime** bằng Chart.js + polling API backend.
- **Thêm** tương tác điều tra:
  - Click từng alert để mở modal điều tra flow.
  - Thanh trượt `Alert threshold` và `Batch size` cập nhật trực tiếp backend.
- **Bổ sung** nhóm AI model metrics: accuracy, F1, recall, inference speed, confidence histogram, FPR.

#### `main_monitor.py`
- **Sửa** lỗi Windows encoding (cp1252) gây vỡ log Unicode.
- **Thêm** `_prepare_batch_for_scaler()` để tự căn chỉnh số chiều feature trước khi scale, tránh lỗi mismatch.
- **Sửa** auto-train từ `os.system(...)` sang `subprocess.run([...])` để không lỗi khi đường dẫn có khoảng trắng trên Windows.

#### `README.md`
- **Cập nhật** tài liệu để phản ánh trạng thái hiện tại:
  - Bổ sung mục `Live SOC Dashboard` và hướng dẫn chạy `python dashboard_server.py`.
  - Làm rõ feature set: CICIDS2017 có 79 features raw, pipeline realtime dùng 18 canonical features.
  - Bổ sung mô tả file mới `dashboard_server.py` và `dashboard_static/index.html`.

### Kết quả xác nhận
- Dashboard backend đã chạy thành công trên localhost.
- API dữ liệu realtime hoạt động, giao diện cập nhật theo thời gian thực.
- Hệ thống tương thích tốt trên Windows (UTF-8 logging + đường dẫn có khoảng trắng).

---

## [2026-04-01] — Sửa lỗi dataset & pipeline training

### Vấn đề phát hiện
1. Dataset CICIDS2017 (8 CSV files) đặt tại `MachineLearningCSV/MachineLearningCVE/`
   nhưng code chỉ trỏ đến `data/dataset.csv` (không tồn tại) → fallback sang synthetic data.
2. Sau khi fix vị trí dataset, số features bị "đội lên" từ 18 lên 78 (toàn bộ cột CICIDS2017)
   → mismatch với `PacketFeatureExtractor` chỉ trích xuất 18 features khi inference.
3. Cột `Fwd Header Length` bị lặp lại 2 lần trong raw CSV (vị trí 35 và 56),
   pandas tự đổi tên thành `Fwd Header Length.1`.
4. `Protocol` có trong `feature_names` của extractor nhưng không có trong CICIDS2017.
5. `train_model.py` crash với `UnicodeEncodeError` trên Windows (cp1252 không encode
   được ký tự box-drawing UTF-8: `╔`, `═`, `╗`, `║`, `╚`, `╝`).

---

### Thay đổi chi tiết

#### `config.py`
- **Thêm** biến `DATASET_DIR = os.path.join(BASE_DIR, "MachineLearningCSV", "MachineLearningCVE")`
  để trỏ đúng đến thư mục chứa 8 CSV files CICIDS2017.
- **Giữ nguyên** `DATASET_PATH = data/dataset.csv` làm đường dẫn merged cache.

#### `dataset_loader.py`
- **Thêm** hằng số `CANONICAL_FEATURES` — danh sách 18 features chuẩn mà
  `PacketFeatureExtractor` có thể trích xuất từ live traffic.
- **Sửa** `load_cicids2017()`: chỉ giữ các cột có trong `CANONICAL_FEATURES`.
- **Thêm** hàm `load_cicids2017_dir(dataset_dir)`: load và merge toàn bộ CSV files.
- **Sửa** `load_dataset()`: thứ tự ưu tiên cache → DATASET_DIR → synthetic.

#### `feature_extractor.py`
- **Bỏ** `Protocol` khỏi `PacketFeatureExtractor.feature_names` (CICIDS2017 không có cột này).

#### `train_model.py`
- **Thêm** UTF-8 encoding fix để chạy trên Windows (cp1252 không hỗ trợ box-drawing chars).

#### `README.md`
- **Viết lại** phản ánh đúng cấu trúc hiện tại, thêm hướng dẫn dataset CICIDS2017.

### Kết quả sau khi sửa

| Chỉ số | Giá trị |
|--------|---------|
| Dataset | CICIDS2017 thực tế — 2,827,876 flows từ 8 files |
| Features | 18 (canonical, nhất quán train ↔ inference) |
| Model | XGBoost + CUDA (RTX 3050 Laptop, profile LOW) |
| Training time | 3.13 giây |
| Accuracy | 99.21% |
| Precision | 96.72% |
| Recall | 99.34% |
| F1-Score | 98.01% |
| Throughput inference | 3,348,911 samples/giây |

---

## [trước 2026-04-01] — Khởi tạo dự án ban đầu

Commit: `7a6d1cb` — "Optimize training pipeline for GPU (RTX 3070 Ti)"

### Files khởi tạo
| File | Mô tả |
|------|-------|
| `config.py` | Cấu hình hệ thống, GPU profile tự động (LOW/MEDIUM/HIGH/ULTRA) |
| `dataset_loader.py` | Load CICIDS2017/UNSW-NB15, synthetic data fallback |
| `feature_extractor.py` | `PacketFeatureExtractor` (packet-based) + `FlowAggregator` (flow-based) |
| `ai_model.py` | `IDSModel` hỗ trợ RandomForest / XGBoost / MLP / PyTorch MLP |
| `train_model.py` | Pipeline train: load → preprocess → train → evaluate → save |
| `main_monitor.py` | Orchestrator Producer-Consumer-AI với threading/multiprocessing |
| `benchmark.py` | Benchmark throughput single vs multi thread/process |
| `requirements.txt` | Dependencies: numpy, pandas, scikit-learn, xgboost, torch (optional) |
