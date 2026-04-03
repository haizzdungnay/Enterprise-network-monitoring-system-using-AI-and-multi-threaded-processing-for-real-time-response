"""
╔══════════════════════════════════════════════════════════════════╗
║      DATASET_LOADER.PY - LOAD & PREPROCESS DATASET IDS         ║
╚══════════════════════════════════════════════════════════════════╝

LÝ THUYẾT: Feature Engineering cho Intrusion Detection
──────────────────────────────────────────────────────
Network traffic được phân tích ở 2 cấp độ:

1. PACKET-LEVEL: Từng gói tin riêng lẻ
   - Header fields: IP src/dst, port, protocol, flags
   - Payload size, TTL, window size
   → Nhanh nhưng thiếu context

2. FLOW-LEVEL: Nhóm các packet cùng 5-tuple
   (src_ip, dst_ip, src_port, dst_port, protocol)
   - Duration, total bytes, total packets
   - Statistical features: mean, std, min, max của packet sizes
   - Flag counts: SYN, ACK, FIN, RST, PSH, URG
   - Inter-arrival time statistics
   → Chính xác hơn, là standard cho IDS hiện đại

Dataset CICIDS2017 và UNSW-NB15 đều sử dụng FLOW-LEVEL features.

PREPROCESSING quan trọng:
──────────────────────────
1. Handle missing values (NaN, Inf)
2. Encode categorical features
3. Normalize/Scale numerical features
4. Handle class imbalance (Normal >> Attack)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import config
import warnings
warnings.filterwarnings('ignore')

# Features mà PacketFeatureExtractor có thể trích xuất từ live traffic.
# Model phải được train trên ĐÚNG tập features này để inference hoạt động.
# Lưu ý: 'Protocol' bị loại vì CICIDS2017 không có cột này.
CANONICAL_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Down/Up Ratio',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'SYN Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Active Mean',
    'Idle Mean',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Destination Port',
]


def generate_synthetic_dataset(n_samples=100000, save_path=None):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  TẠO DATASET GIẢ LẬP DẠNG CICIDS2017                ║
    ╠════════════════════════════════════════════════════════╣
    ║  Dùng khi không có dataset thật để demo & benchmark.  ║
    ║  Features mô phỏng đặc trưng thực tế của traffic.    ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Tại sao cần synthetic data?
    ───────────────────────────────────────
    - Dataset thật (CICIDS2017) rất lớn (~8GB CSV)
    - Không phải lúc nào cũng download được
    - Synthetic data giúp test pipeline nhanh
    - Đảm bảo code hoạt động trước khi dùng data thật
    """
    np.random.seed(config.RANDOM_STATE)

    # ── Tỷ lệ normal vs attack (giống thực tế: ~80% normal) ──
    n_normal = int(n_samples * 0.8)
    n_attack = n_samples - n_normal

    # ── NORMAL TRAFFIC patterns ──
    # Flow Duration (microseconds): Thường ngắn cho web browsing
    normal_duration = np.random.exponential(500000, n_normal)
    # Total Forward Packets: Ít packet cho request thông thường
    normal_fwd_packets = np.random.poisson(10, n_normal)
    # Total Backward Packets: Response packets
    normal_bwd_packets = np.random.poisson(8, n_normal)
    # Flow Bytes/s: Bandwidth usage bình thường
    normal_flow_bytes = np.random.lognormal(8, 2, n_normal)
    # Flow Packets/s: Packet rate bình thường
    normal_flow_packets = np.random.lognormal(3, 1, n_normal)
    # Down/Up Ratio: Download thường > Upload
    normal_down_up = np.random.uniform(0.5, 5.0, n_normal)
    # Fwd Packet Length Mean
    normal_fwd_len = np.random.normal(200, 100, n_normal).clip(0)
    # Bwd Packet Length Mean
    normal_bwd_len = np.random.normal(500, 200, n_normal).clip(0)
    # SYN Flag Count: Bình thường chỉ có 1 SYN per flow
    normal_syn = np.random.choice([0, 1], n_normal, p=[0.3, 0.7])
    # FIN Flag Count
    normal_fin = np.random.choice([0, 1], n_normal, p=[0.4, 0.6])
    # RST Flag Count: Hiếm trong traffic bình thường
    normal_rst = np.random.choice([0, 1], n_normal, p=[0.9, 0.1])
    # PSH Flag Count
    normal_psh = np.random.poisson(2, n_normal)
    # ACK Flag Count
    normal_ack = np.random.poisson(5, n_normal)
    # Active/Idle Mean
    normal_active = np.random.exponential(100, n_normal)
    normal_idle = np.random.exponential(1000, n_normal)
    # IAT (Inter-Arrival Time) Mean
    normal_iat_mean = np.random.exponential(50000, n_normal)
    normal_iat_std = np.random.exponential(30000, n_normal)
    # Destination Port
    normal_dst_port = np.random.choice([80, 443, 53, 22, 8080, 3306],
                                        n_normal, p=[0.3, 0.35, 0.15, 0.1, 0.05, 0.05])

    # ── ATTACK TRAFFIC patterns ──
    """
    LÝ THUYẾT: Đặc trưng của tấn công mạng
    ────────────────────────────────────────
    - DDoS: Rất nhiều packet, duration ngắn, nhiều SYN flags
    - Port Scan: Nhiều flow đến nhiều port khác nhau, ít packet/flow
    - Brute Force: Nhiều connection attempts, duration dài
    - Data Exfiltration: Upload >> Download (bất thường)
    """
    attack_duration = np.random.exponential(1000000, n_attack)
    attack_fwd_packets = np.random.poisson(50, n_attack)       # Nhiều packet hơn
    attack_bwd_packets = np.random.poisson(3, n_attack)        # Ít response
    attack_flow_bytes = np.random.lognormal(10, 3, n_attack)   # Bytes bất thường
    attack_flow_packets = np.random.lognormal(5, 2, n_attack)  # Rate cao
    attack_down_up = np.random.uniform(0.01, 0.5, n_attack)    # Upload nhiều
    attack_fwd_len = np.random.normal(100, 50, n_attack).clip(0)
    attack_bwd_len = np.random.normal(50, 30, n_attack).clip(0)
    attack_syn = np.random.choice([0, 1, 2, 5, 10], n_attack,
                                   p=[0.1, 0.2, 0.3, 0.2, 0.2])  # Nhiều SYN
    attack_fin = np.random.choice([0, 1], n_attack, p=[0.7, 0.3])
    attack_rst = np.random.choice([0, 1, 2], n_attack, p=[0.3, 0.4, 0.3])  # Nhiều RST
    attack_psh = np.random.poisson(8, n_attack)
    attack_ack = np.random.poisson(15, n_attack)
    attack_active = np.random.exponential(500, n_attack)
    attack_idle = np.random.exponential(200, n_attack)
    attack_iat_mean = np.random.exponential(10000, n_attack)   # IAT nhỏ (packet dồn dập)
    attack_iat_std = np.random.exponential(5000, n_attack)
    attack_dst_port = np.random.choice([80, 443, 22, 23, 445, 3389],
                                        n_attack, p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.15])

    # ── Gộp data ──
    features = {
        'Flow Duration': np.concatenate([normal_duration, attack_duration]),
        'Total Fwd Packets': np.concatenate([normal_fwd_packets, attack_fwd_packets]),
        'Total Backward Packets': np.concatenate([normal_bwd_packets, attack_bwd_packets]),
        'Flow Bytes/s': np.concatenate([normal_flow_bytes, attack_flow_bytes]),
        'Flow Packets/s': np.concatenate([normal_flow_packets, attack_flow_packets]),
        'Down/Up Ratio': np.concatenate([normal_down_up, attack_down_up]),
        'Fwd Packet Length Mean': np.concatenate([normal_fwd_len, attack_fwd_len]),
        'Bwd Packet Length Mean': np.concatenate([normal_bwd_len, attack_bwd_len]),
        'SYN Flag Count': np.concatenate([normal_syn, attack_syn]),
        'FIN Flag Count': np.concatenate([normal_fin, attack_fin]),
        'RST Flag Count': np.concatenate([normal_rst, attack_rst]),
        'PSH Flag Count': np.concatenate([normal_psh, attack_psh]),
        'ACK Flag Count': np.concatenate([normal_ack, attack_ack]),
        'Active Mean': np.concatenate([normal_active, attack_active]),
        'Idle Mean': np.concatenate([normal_idle, attack_idle]),
        'Flow IAT Mean': np.concatenate([normal_iat_mean, attack_iat_mean]),
        'Flow IAT Std': np.concatenate([normal_iat_std, attack_iat_std]),
        'Destination Port': np.concatenate([normal_dst_port, attack_dst_port]),
    }

    labels = np.concatenate([
        np.zeros(n_normal, dtype=int),    # 0 = Normal
        np.ones(n_attack, dtype=int),     # 1 = Attack
    ])

    df = pd.DataFrame(features)
    df['Label'] = labels

    # Shuffle
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[+] Đã tạo synthetic dataset: {save_path}")
        print(f"    Tổng: {len(df)} | Normal: {n_normal} | Attack: {n_attack}")

    return df


def load_cicids2017(filepath):
    """
    Load dataset CICIDS2017 từ file CSV.

    LÝ THUYẾT: CICIDS2017 Structure
    ────────────────────────────────
    - Tạo bởi Canadian Institute for Cybersecurity
    - 7 file CSV tương ứng 5 ngày capture
    - 80+ features từ CICFlowMeter tool
    - Labels: BENIGN, DoS, DDoS, PortScan, BruteForce, ...
    """
    print(f"[*] Loading CICIDS2017 tu {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Clean column names (CICIDS2017 thuong co spaces thua)
    df.columns = df.columns.str.strip()

    # Label column
    label_col = 'Label' if 'Label' in df.columns else 'label'

    # Binary classification: BENIGN = 0, con lai = 1
    if df[label_col].dtype == 'object':
        df['Label_Binary'] = (df[label_col].str.strip() != 'BENIGN').astype(int)
    else:
        df['Label_Binary'] = df[label_col].astype(int)

    # Chi giu cac features co trong CANONICAL_FEATURES va co trong CSV
    # (loai bo duplicate 'Fwd Header Length.1' va cac cols khong can thiet)
    feature_cols = [c for c in CANONICAL_FEATURES if c in df.columns]
    missing = [c for c in CANONICAL_FEATURES if c not in df.columns]
    if missing:
        print(f"    [!] Features khong co trong CSV (bo qua): {missing}")

    # Handle Inf, NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    # Convert to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_cols)

    print(f"    Loaded {len(df):,} flows | Features: {len(feature_cols)}")
    return df[feature_cols], df['Label_Binary'], feature_cols


def load_unsw_nb15(filepath):
    """
    Load dataset UNSW-NB15 từ file CSV.

    LÝ THUYẾT: UNSW-NB15 Structure
    ───────────────────────────────
    - Tạo bởi UNSW Canberra
    - 49 features (bao gồm cả flow và content features)
    - 9 loại tấn công: Fuzzers, Analysis, Backdoors, DoS, Exploits,
      Generic, Reconnaissance, Shellcode, Worms
    - Label column: 0 = Normal, 1 = Attack
    """
    print(f"[*] Loading UNSW-NB15 từ {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = 'label' if 'label' in df.columns else 'Label'

    # Loại bỏ columns không phải feature
    exclude = [label_col, 'id', 'attack_cat', 'srcip', 'dstip']
    feature_cols = [c for c in df.columns if c not in exclude]

    # Encode categorical columns
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Handle Inf, NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    print(f"    Loaded {len(df)} records | Features: {len(feature_cols)}")
    return df[feature_cols], df[label_col].astype(int), feature_cols


def preprocess_data(X, y, feature_names):
    """
    ╔════════════════════════════════════════════════════════╗
    ║  PREPROCESSING PIPELINE                                ║
    ╠════════════════════════════════════════════════════════╣
    ║  1. Train/Test split (stratified)                     ║
    ║  2. Feature scaling (StandardScaler)                  ║
    ║  3. Save scaler & feature names cho inference         ║
    ╚════════════════════════════════════════════════════════╝

    LÝ THUYẾT: Tại sao cần Scaling?
    ────────────────────────────────
    - Các features có scale rất khác nhau:
      + Flow Duration: 0 ~ 120,000,000 (microseconds)
      + Packet Count: 0 ~ 1000
      + Protocol: 1, 6, 17
    - Nếu không scale, features có giá trị lớn sẽ dominate
    - StandardScaler: (x - mean) / std → mean=0, std=1
    - LƯU Ý: Fit scaler chỉ trên TRAIN set, transform cả train+test
      (tránh data leakage)
    """
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_array = y.values if hasattr(y, 'values') else np.array(y)

    # Stratified split: giữ tỷ lệ class giống nhau ở train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_array
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit + Transform
    X_test_scaled = scaler.transform(X_test)          # Chỉ Transform!

    # Save scaler và feature names cho inference
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(feature_names, config.FEATURE_NAMES_PATH)

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Normal: {sum(y_train==0)}/{sum(y_test==0)} | "
          f"Attack: {sum(y_train==1)}/{sum(y_test==1)}")
    print(f"    Scaler saved: {config.SCALER_PATH}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_cicids2017_dir(dataset_dir):
    """
    Load và merge toàn bộ CSV files trong thư mục CICIDS2017.

    CICIDS2017 gồm 8 file tương ứng 5 ngày làm việc:
      Monday    : Traffic bình thường
      Tuesday   : Brute Force (FTP-Patator, SSH-Patator)
      Wednesday : DoS/DDoS (Slowloris, Slowhttptest, Hulk, GoldenEye)
      Thursday  : Web Attacks + Infiltration
      Friday    : PortScan, DDoS (LOIT)
    """
    import glob
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
    if not csv_files:
        return None, None, None

    print(f"[*] Tìm thấy {len(csv_files)} file CSV trong {dataset_dir}")
    dfs = []
    for f in csv_files:
        fname = os.path.basename(f)
        print(f"    → Đang load: {fname}")
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[+] Merged: {len(combined):,} flows từ {len(csv_files)} files")

    label_col = 'Label' if 'Label' in combined.columns else 'label'
    combined['Label_Binary'] = (combined[label_col].str.strip() != 'BENIGN').astype(int)

    # Chỉ giữ CANONICAL_FEATURES (loại bỏ duplicate Fwd Header Length.1, v.v.)
    feature_cols = [c for c in CANONICAL_FEATURES if c in combined.columns]
    missing = [c for c in CANONICAL_FEATURES if c not in combined.columns]
    if missing:
        print(f"    [!] Features không có trong CSV (bỏ qua): {missing}")

    combined[feature_cols] = combined[feature_cols].replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=feature_cols)
    for col in feature_cols:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined = combined.dropna(subset=feature_cols)

    print(f"    Sau clean: {len(combined):,} flows | Features: {len(feature_cols)}")
    return combined[feature_cols], combined['Label_Binary'], feature_cols


def load_cicids2017_folder(folder_path, sample_per_file=None):
    """
    Load và merge tất cả file CSV CICIDS2017 từ một folder, có stratified sampling.

    LÝ THUYẾT: Tại sao sample?
    ──────────────────────────
    CICIDS2017 có ~2.8M rows, tổng ~847MB CSV.
    Load hết cần ~6-8GB RAM. Sample 100K/file (800K tổng)
    vẫn đại diện đủ mọi loại tấn công, train vài phút trên GPU.
    """
    import glob
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    csv_files = [f for f in csv_files
                 if os.path.basename(f) != 'dataset.csv']

    if not csv_files:
        return None, None, None

    print(f"[*] Tìm thấy {len(csv_files)} file CSV trong {folder_path}")
    chunks = []
    label_stats = {}

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, low_memory=False,
                             encoding='utf-8', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(fpath, low_memory=False,
                             encoding='latin-1', on_bad_lines='skip')

        df.columns = df.columns.str.strip()

        # Tìm cột label với các biến thể tên khác nhau
        label_col = None
        for candidate in ['Label', 'label', 'LABEL']:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            print(f"    [!] Bỏ qua {fname}: không có cột Label")
            continue
        if label_col != 'Label':
            df = df.rename(columns={label_col: 'Label'})

        df['Label'] = df['Label'].astype(str).str.strip()

        # Sample giữ đúng tỷ lệ class trong từng file (stratified)
        if sample_per_file and len(df) > sample_per_file:
            sampled_parts = []
            for _, grp in df.groupby('Label'):
                n = max(1, int(sample_per_file * len(grp) / len(df)))
                sampled_parts.append(grp.sample(min(len(grp), n),
                                                random_state=config.RANDOM_STATE))
            df = pd.concat(sampled_parts, ignore_index=True)

        for lbl, cnt in df['Label'].value_counts().items():
            label_stats[lbl] = label_stats.get(lbl, 0) + cnt

        chunks.append(df)
        print(f"    {fname[:55]:<55} {len(df):>7,} rows")

    if not chunks:
        return None, None, None

    df_all = pd.concat(chunks, ignore_index=True)
    df_all = df_all.sample(frac=1,
                           random_state=config.RANDOM_STATE).reset_index(drop=True)

    print(f"\n    Tổng: {len(df_all):,} flows")
    print(f"    Phân bố label:")
    for lbl, cnt in sorted(label_stats.items(), key=lambda x: -x[1]):
        print(f"      {str(lbl)[:35]:<35} {cnt:>8,}  ({cnt/len(df_all)*100:.1f}%)")

    # Binary: BENIGN=0, mọi loại attack=1
    df_all['Label_Binary'] = (
        df_all['Label'].str.strip() != 'BENIGN'
    ).astype(int)

    # Chỉ giữ CANONICAL_FEATURES để nhất quán với PacketFeatureExtractor
    feature_cols = [c for c in CANONICAL_FEATURES if c in df_all.columns]
    missing = [c for c in CANONICAL_FEATURES if c not in df_all.columns]
    if missing:
        print(f"    [!] Features không có trong CSV (bỏ qua): {missing}")

    df_all[feature_cols] = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    df_all = df_all.dropna(subset=feature_cols)

    print(f"    Sau cleanup: {len(df_all):,} flows | {len(feature_cols)} features")
    return df_all[feature_cols], df_all['Label_Binary'], feature_cols


def load_dataset():
    """
    Entry point: Load dataset theo thứ tự ưu tiên:
      1. Merged cache (dataset.csv) nếu đã tồn tại → load nhanh
      2. Folder nhiều file CSV CICIDS2017 (DATASET_DIR) → merge + stratified sample + lưu cache
      3. Synthetic data → fallback khi không có gì
    """
    # Ưu tiên 1: Merged cache (tránh phải load lại 8 file mỗi lần)
    if os.path.exists(config.DATASET_PATH):
        print(f"[*] Dùng merged cache: {config.DATASET_PATH}")
        if config.DATASET_TYPE == "cicids2017":
            X, y, names = load_cicids2017(config.DATASET_PATH)
        else:
            X, y, names = load_unsw_nb15(config.DATASET_PATH)
        return preprocess_data(X, y, names)

    # Ưu tiên 2: Thư mục CICIDS2017 thực tế
    if (config.DATASET_TYPE == "cicids2017"
            and hasattr(config, 'DATASET_DIR')
            and os.path.isdir(config.DATASET_DIR)):
        sample_n = getattr(config, 'CICIDS_SAMPLE_PER_FILE', 100000)
        X, y, names = load_cicids2017_folder(config.DATASET_DIR,
                                             sample_per_file=sample_n)
        if X is not None:
            # Lưu merged cache để lần sau load nhanh hơn
            print(f"[*] Đang lưu merged cache → {config.DATASET_PATH}")
            df_cache = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X, columns=names)
            df_cache['Label'] = y.values if hasattr(y, 'values') else y
            df_cache.to_csv(config.DATASET_PATH, index=False)
            print(f"[+] Cache đã lưu: {config.DATASET_PATH}")
            return preprocess_data(X, y, names)
        else:
            print("[!] Không tìm thấy CSV trong DATASET_DIR. Tạo synthetic data...")

    # Ưu tiên 3: Synthetic data (demo / unit test)
    print("[!] Không tìm thấy dataset. Tạo synthetic data...")
    df = generate_synthetic_dataset(n_samples=100000, save_path=config.DATASET_PATH)
    feature_cols = [c for c in df.columns if c != 'Label']
    return preprocess_data(df[feature_cols], df['Label'], feature_cols)


if __name__ == "__main__":
    # Test loading
    X_train, X_test, y_train, y_test, scaler = load_dataset()
    print(f"\n[OK] Dataset loaded successfully!")
    print(f"     X_train shape: {X_train.shape}")
    print(f"     X_test shape:  {X_test.shape}")
