"""
pipeline.py
-----------
Orchestrates the full churn prediction pipeline:

    [1] Data Generation
    [2] Feature Engineering
    [3] Model Training   (6 models compared via MLflow)
    [4] Customer Segmentation  (K-Means k=5)
    [5] A/B Test Simulation    (causal inference + CUPED)
    [6] Dashboard Build

Run:
    python pipeline.py [--skip-data]
"""

import argparse
import time
import os


def banner(step: int, msg: str):
    print(f"\n{'='*65}")
    print(f"  STEP {step}: {msg}")
    print(f"{'='*65}\n")


def run_pipeline(skip_data: bool = False):
    start = time.time()
    os.makedirs("data",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    if not skip_data or not os.path.exists("data/customers_raw.parquet"):
        banner(1, "Data Generation")
        from data_generation import generate_customers
        df = generate_customers()
        df.to_parquet("data/customers_raw.parquet", index=False)
        print(f"[DONE] {len(df):,} customer records generated.")
    else:
        import pandas as pd
        df = pd.read_parquet("data/customers_raw.parquet")
        print(f"[SKIP] Using existing data ({len(df):,} rows).")

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    banner(2, "Feature Engineering")
    import pandas as pd
    from feature_engineering import build_features
    df = pd.read_parquet("data/customers_raw.parquet")
    df, le_map = build_features(df)
    df.to_parquet("data/customers_features.parquet", index=False)
    print(f"[DONE] Feature matrix: {df.shape}")

    # ── Step 3: Model Training ────────────────────────────────────────────────
    banner(3, "Model Training (6 models, MLflow comparison)")
    from model_training import train
    best_model, best_threshold, best_name = train("data/customers_features.parquet")
    print(f"[DONE] Best model: {best_name}  Threshold: {best_threshold:.4f}")

    # ── Step 4: Customer Segmentation ─────────────────────────────────────────
    banner(4, "Customer Segmentation (K-Means k=5)")
    from segmentation import run_segmentation
    df_seg, profile = run_segmentation("data/customers_features.parquet")
    print(f"[DONE] {profile['n_customers'].sum():,} customers segmented.")

    # ── Step 5: A/B Test ──────────────────────────────────────────────────────
    banner(5, "A/B Testing & Causal Inference (CUPED)")
    from ab_testing import run_ab_test
    ate_raw, ate_cuped, hte_df = run_ab_test("data/customers_features.parquet")
    print(f"[DONE] ATE = {ate_raw['ate']*100:.2f}pp  "
          f"(p = {ate_raw['p_value']:.6f})")

    # ── Step 6: Dashboard ─────────────────────────────────────────────────────
    banner(6, "Dashboard Generation")
    from dashboard import build_dashboard
    build_dashboard()
    print("[DONE] Dashboard → outputs/churn_dashboard.png")

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*65}")
    print("\nKey Outputs:")
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(f"outputs/{f}") // 1024
        print(f"  outputs/{f}  ({size} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline"
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation if parquet exists")
    args = parser.parse_args()
    run_pipeline(skip_data=args.skip_data)
