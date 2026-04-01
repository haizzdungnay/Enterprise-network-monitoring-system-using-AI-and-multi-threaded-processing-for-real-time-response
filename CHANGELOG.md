# CHANGELOG — Hệ Thống Giám Sát Mạng Doanh Nghiệp

Ghi lại toàn bộ thay đổi theo thứ tự thời gian.
Format: `[FILE] Mô tả thay đổi`

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
- **Giữ nguyên** `DATASET_PATH = data/dataset.csv` làm đường dẫn merged cache
  (tự động tạo lần đầu chạy, tránh phải load lại 8 file mỗi lần).

---

#### `dataset_loader.py`
- **Thêm** hằng số `CANONICAL_FEATURES` — danh sách 18 features chuẩn mà
  `PacketFeatureExtractor` có thể trích xuất từ live traffic:
  ```
  Flow Duration, Total Fwd Packets, Total Backward Packets,
  Flow Bytes/s, Flow Packets/s, Down/Up Ratio,
  Fwd Packet Length Mean, Bwd Packet Length Mean,
  SYN Flag Count, FIN Flag Count, RST Flag Count,
  PSH Flag Count, ACK Flag Count,
  Active Mean, Idle Mean,
  Flow IAT Mean, Flow IAT Std,
  Destination Port
  ```
  Mục đích: đảm bảo training và inference dùng đúng tập features.

- **Sửa** `load_cicids2017()`: thay vì lấy toàn bộ cột CSV (78 features),
  giờ chỉ giữ các cột có trong `CANONICAL_FEATURES`.
  Tự động bỏ `Fwd Header Length.1` (duplicate) và các cột không liên quan.

- **Thêm** hàm `load_cicids2017_dir(dataset_dir)`:
  Load và merge toàn bộ CSV files trong thư mục CICIDS2017 thành 1 DataFrame.
  Áp dụng cùng logic filter `CANONICAL_FEATURES`.

- **Sửa** `load_dataset()` — cập nhật thứ tự ưu tiên 3 tầng:
  1. Dùng `data/dataset.csv` nếu đã có (merged cache, load nhanh).
  2. Load từ `DATASET_DIR` (8 CSV files), merge, lưu cache vào `data/dataset.csv`.
  3. Tạo synthetic data 100K flows nếu không tìm thấy gì (chỉ dùng để demo/test).

- **Thêm** hàm helper `_make_synthetic()` để tách logic synthetic data ra riêng.

---

#### `feature_extractor.py`
- **Bỏ** `Protocol` khỏi `PacketFeatureExtractor.feature_names`.
  Lý do: CICIDS2017 không có cột Protocol → không thể train trên feature này.
  Kết quả: giảm từ 19 xuống 18 features, khớp với `CANONICAL_FEATURES`.

- **Cập nhật** `extract_from_packet_dict()`:
  - Bỏ dòng `features[17] = packet_info.get('protocol', 6)`
  - Chuyển `features[18] = packet_info.get('dst_port', 0)`
    thành `features[17] = packet_info.get('dst_port', 0)`

- **Cập nhật** `generate_simulated_features()`:
  - Bỏ `np.random.choice([6, 17, 1])` (Protocol) khỏi cả 2 nhánh attack/normal.
  - Thêm comment rõ tên feature cho từng phần tử mảng (18 elements).

---

#### `train_model.py`
- **Thêm** UTF-8 encoding fix ở đầu file để chạy được trên Windows:
  ```python
  import sys
  if hasattr(sys.stdout, 'reconfigure'):
      sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  ```
  Lý do: Windows console mặc định dùng cp1252, không encode được ký tự
  box-drawing Unicode (`╔═╗║╚╝─`) dẫn đến `UnicodeEncodeError` khi chạy.

---

#### `README.md`
- **Viết lại hoàn toàn** để phản ánh đúng cấu trúc và tính năng hiện tại:
  - Cập nhật mục "Cài Đặt & Chạy": hướng dẫn đặt dataset tại `MachineLearningCSV/MachineLearningCVE/`.
  - Cập nhật cấu trúc thư mục: bỏ `producer.py`, `consumer.py`, `ai_worker.py`
    (không tồn tại — các chức năng này nằm trong `main_monitor.py`).
  - Thêm mục "Dataset CICIDS2017": bảng chi tiết 8 file theo ngày/loại tấn công,
    sơ đồ pipeline load 3 tầng ưu tiên.
  - Thêm mục "GPU & Cấu hình": bảng GPU profiles (LOW/MEDIUM/HIGH/ULTRA),
    bảng tham số `config.py` bao gồm `DATASET_DIR` mới.
  - Sửa "80+ features" thành "79 features" (thực tế của CICIDS2017 CICFlowMeter).

---

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
| `feature_extractor.py` | Trích xuất features từ packet/flow thực tế |
| `requirements.txt` | Dependencies: numpy, pandas, scikit-learn, xgboost, torch (optional) |
