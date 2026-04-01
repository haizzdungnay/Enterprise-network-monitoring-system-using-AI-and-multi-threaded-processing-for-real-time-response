"""
╔══════════════════════════════════════════════════════════════════╗
║         TRAIN_MODEL.PY - SCRIPT TRAIN MODEL IDS                ║
╚══════════════════════════════════════════════════════════════════╝

Chạy: python train_model.py

Quy trình:
  1. Load dataset (thật hoặc synthetic)
  2. Preprocessing (scale, split)
  3. Train model
  4. Evaluate trên test set
  5. Benchmark batch sizes
  6. Save model + scaler
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import os
import config
from dataset_loader import load_dataset
from ai_model import IDSModel, benchmark_single_vs_batch, HAS_TORCH


def main():
    print("╔" + "═" * 58 + "╗")
    print("║     TRAINING IDS MODEL - Hệ Thống Giám Sát Mạng        ║")
    print("╚" + "═" * 58 + "╝")

    # GPU warmup: khởi tạo CUDA context trước để đo training time chính xác
    if config.USE_GPU and HAS_TORCH:
        import torch
        print(f"[*] GPU Warmup: {torch.cuda.get_device_name(0)}")
        _dummy = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        del _dummy
        print(f"    VRAM total:  {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        print(f"    VRAM free:   {torch.cuda.mem_get_info()[0] // 1024**2} MB")

    config.print_config()

    # ── Bước 1: Load & Preprocess Data ──
    print("\n" + "─" * 60)
    print("  BƯỚC 1: Load & Preprocess Dataset")
    print("─" * 60)
    X_train, X_test, y_train, y_test, scaler = load_dataset()

    # ── Bước 2: Train Model ──
    print("\n" + "─" * 60)
    print("  BƯỚC 2: Train Model")
    print("─" * 60)
    model = IDSModel(config.MODEL_TYPE)
    model.feature_names = list(
        __import__('joblib').load(config.FEATURE_NAMES_PATH)
    )
    train_time = model.train(X_train, y_train)

    # ── Bước 3: Evaluate ──
    print("\n" + "─" * 60)
    print("  BƯỚC 3: Evaluate Model")
    print("─" * 60)
    metrics = model.evaluate(X_test, y_test)

    # ── Bước 4: Benchmark Batch Sizes ──
    print("\n" + "─" * 60)
    print("  BƯỚC 4: Benchmark Batch Prediction")
    print("─" * 60)
    batch_results = benchmark_single_vs_batch(model, X_test)

    # ── Bước 5: Save Model ──
    print("\n" + "─" * 60)
    print("  BƯỚC 5: Save Model & Results")
    print("─" * 60)
    model.save()

    # Save results
    results = {
        'model_type': config.MODEL_TYPE,
        'train_time': train_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'throughput': metrics['throughput']
        },
        'batch_benchmark': {
            str(k): v for k, v in batch_results.items()
        }
    }

    results_path = os.path.join(config.RESULTS_DIR, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results saved: {results_path}")

    print("\n" + "═" * 60)
    print("  TRAINING COMPLETE!")
    print("═" * 60)


if __name__ == "__main__":
    main()
