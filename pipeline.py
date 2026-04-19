"""
pipeline.py
-----------
Orchestrates the full end-to-end fraud detection pipeline:

    [1] Data Generation
    [2] Feature Engineering
    [3] Model Training  (XGBoost + SMOTE)
    [4] Model Evaluation (SHAP + metrics)
    [5] Dashboard Build

Run:
    python pipeline.py [--skip-data]
"""

import argparse
import time
import os
import sys


def banner(step: int, msg: str):
    print(f"\n{'='*65}")
    print(f"  STEP {step}: {msg}")
    print(f"{'='*65}\n")


def run_pipeline(skip_data: bool = False):
    start = time.time()

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    if not skip_data or not os.path.exists("data/transactions_raw.parquet"):
        banner(1, "Data Generation")
        from data_generation import generate_transactions, add_velocity_features
        import os as _os; _os.makedirs("data", exist_ok=True)
        df = generate_transactions()
        df = add_velocity_features(df)
        df.to_parquet("data/transactions_raw.parquet", index=False)
        print(f"[DONE] {len(df):,} rows generated.")
    else:
        print("[SKIP] Data already exists — skipping generation.")

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    banner(2, "Feature Engineering")
    import pandas as pd
    from feature_engineering import build_features
    df = pd.read_parquet("data/transactions_raw.parquet")
    df = build_features(df)
    df.to_parquet("data/transactions_features.parquet", index=False)
    print(f"[DONE] Feature matrix: {df.shape}")

    # ── Step 3: Model Training ────────────────────────────────────────────────
    banner(3, "Model Training (XGBoost + SMOTE)")
    from model_training import train
    model, threshold = train("data/transactions_features.parquet")
    print(f"[DONE] Model trained. Threshold: {threshold:.4f}")

    # ── Step 4: Model Evaluation ──────────────────────────────────────────────
    banner(4, "Model Evaluation + SHAP")
    from model_evaluation import evaluate
    evaluate()
    print("[DONE] Evaluation outputs written to outputs/")

    # ── Step 5: Dashboard ─────────────────────────────────────────────────────
    banner(5, "Dashboard Generation")
    from dashboard import build_dashboard
    build_dashboard()
    print("[DONE] Dashboard written to outputs/fraud_dashboard.png")

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*65}")
    print("\nOutputs:")
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(f"outputs/{f}") // 1024
        print(f"  outputs/{f}  ({size} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data generation if parquet already exists"
    )
    args = parser.parse_args()
    run_pipeline(skip_data=args.skip_data)
