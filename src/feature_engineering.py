"""
Feature Engineering & Customer-level RFM Mart Builder
File: src/feature_engineering.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

def build_rfm_features(raw_csv_path: str, output_parquet_path: str, summary_json_path: str):
    print("=" * 65)
    print(" [STEP 4] Feature Engineering & Customer-Level RFM Mart Builder")
    print("=" * 65)

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Input file not found: {raw_csv_path}")

    print(f"Loading raw transaction dataset: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, encoding="utf-8-sig")
    print(f"Loaded {len(df):,} transactions across {df['customer_id'].nunique():,} unique customers.")

    # 1. Aggregation by Customer ID
    print("\n1. Aggregating transaction metrics by customer_id...")
    
    # Base reference year for recency
    REFERENCE_YEAR = 2025

    # Groupby aggregations
    cust_agg = df.groupby("customer_id").agg(
        frequency=("customer_seq", "count"),
        max_seq=("customer_seq", "max"),
        monetary_total=("transaction_amt", "sum"),
        monetary_avg=("transaction_amt", "mean"),
        last_year=("transaction_last_year", "max"),
        start_year=("transaction_start_year", "min"),
        # First demographic observations per customer
        customer_type=("customer_type", "first"),
        age=("age", "first"),
        gender=("gender", "first"),
        region=("region", "first"),
        income_bracket=("income_bracket", "first"),
        occupation_group=("occupation_group", "first"),
        marital_status=("marital_status", "first")
    ).reset_index()

    # Recency & Tenure calculations
    cust_agg["recency_years"] = REFERENCE_YEAR - cust_agg["last_year"]
    # Handle any negative if last_year > 2025 (safeguard)
    cust_agg["recency_years"] = cust_agg["recency_years"].clip(lower=0)
    cust_agg["tenure_years"] = REFERENCE_YEAR - cust_agg["start_year"]
    cust_agg["tenure_years"] = cust_agg["tenure_years"].clip(lower=0)
    cust_agg["is_repeat_customer"] = (cust_agg["frequency"] > 1).astype(int)

    print(f" - Aggregated customer shape: {cust_agg.shape}")

    # 2. Ordinal & Binary Categorical Encodings
    print("\n2. Applying profile feature encodings...")

    age_mapping = {
        "20대": 1,
        "30대": 2,
        "40대": 3,
        "50대": 4,
        "60대": 5,
        "70대": 6,
        "70대 이상": 6,
        "80대 이상": 7
    }
    cust_agg["age_ordinal"] = cust_agg["age"].map(age_mapping).fillna(3).astype(int)

    income_mapping = {
        "1000만원미만": 1,
        "3000만원미만": 2,
        "5000만원미만": 3,
        "8000만원미만": 4,
        "10000만원미만": 5,
        "30000만원미만": 6,
        "10000만원이상": 7
    }
    cust_agg["income_ordinal"] = cust_agg["income_bracket"].map(income_mapping).fillna(2).astype(int)

    cust_agg["gender_is_male"] = (cust_agg["gender"] == "남").astype(int)
    cust_agg["marital_is_married"] = (cust_agg["marital_status"] == "기혼").astype(int)
    cust_agg["sector_is_securities"] = (cust_agg["customer_type"] == "증권").astype(int)
    cust_agg["sector_is_insurance"] = (cust_agg["customer_type"] == "보험").astype(int)

    # 3. Log Transformations for Highly Skewed Distributions
    print("\n3. Applying log1p transformations for monetary, frequency, and tenure...")
    cust_agg["monetary_log"] = np.log1p(cust_agg["monetary_total"])
    cust_agg["monetary_avg_log"] = np.log1p(cust_agg["monetary_avg"])
    cust_agg["frequency_log"] = np.log1p(cust_agg["frequency"])
    cust_agg["recency_log"] = np.log1p(cust_agg["recency_years"])
    cust_agg["tenure_log"] = np.log1p(cust_agg["tenure_years"])

    # 4. Standard & Robust Scaling for Modeling Features
    print("\n4. Fitting StandardScaler and RobustScaler for clustering features...")
    feature_cols = [
        "recency_log",
        "frequency_log",
        "monetary_log",
        "tenure_log",
        "age_ordinal",
        "income_ordinal"
    ]

    std_scaler = StandardScaler()
    scaled_std = std_scaler.fit_transform(cust_agg[feature_cols])
    for idx, col in enumerate(feature_cols):
        cust_agg[f"{col}_std"] = scaled_std[:, idx]

    robust_scaler = RobustScaler()
    scaled_robust = robust_scaler.fit_transform(cust_agg[feature_cols])
    for idx, col in enumerate(feature_cols):
        cust_agg[f"{col}_robust"] = scaled_robust[:, idx]

    # 5. Save to Parquet
    print(f"\n5. Saving processed feature mart to Parquet...")
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    cust_agg.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    file_size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)
    print(f" - Successfully saved: {output_parquet_path} ({file_size_mb:.2f} MB)")

    # 6. Generate Summary JSON
    print(f"\n6. Generating feature mart summary...")
    summary_data = {
        "total_unique_customers": len(cust_agg),
        "columns_count": len(cust_agg.columns),
        "column_names": list(cust_agg.columns),
        "rfm_raw_stats": {
            "recency_years": {
                "mean": float(cust_agg["recency_years"].mean()),
                "median": float(cust_agg["recency_years"].median()),
                "min": float(cust_agg["recency_years"].min()),
                "max": float(cust_agg["recency_years"].max()),
                "zero_recency_pct": float((cust_agg["recency_years"] == 0).mean() * 100)
            },
            "frequency": {
                "mean": float(cust_agg["frequency"].mean()),
                "median": float(cust_agg["frequency"].median()),
                "min": float(cust_agg["frequency"].min()),
                "max": float(cust_agg["frequency"].max()),
                "repeat_customer_pct": float(cust_agg["is_repeat_customer"].mean() * 100)
            },
            "monetary_total_krw": {
                "mean": float(cust_agg["monetary_total"].mean()),
                "median": float(cust_agg["monetary_total"].median()),
                "min": float(cust_agg["monetary_total"].min()),
                "max": float(cust_agg["monetary_total"].max())
            }
        },
        "log_transformed_stats": {
            "monetary_log_mean": float(cust_agg["monetary_log"].mean()),
            "monetary_log_std": float(cust_agg["monetary_log"].std()),
            "frequency_log_mean": float(cust_agg["frequency_log"].mean()),
            "recency_log_mean": float(cust_agg["recency_log"].mean())
        },
        "demographic_aggregates": {
            "male_pct": float(cust_agg["gender_is_male"].mean() * 100),
            "married_pct": float(cust_agg["marital_is_married"].mean() * 100),
            "securities_sector_pct": float(cust_agg["sector_is_securities"].mean() * 100),
            "insurance_sector_pct": float(cust_agg["sector_is_insurance"].mean() * 100)
        }
    }

    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f" - Summary saved: {summary_json_path}")
    print("=" * 65)
    print(" [STEP 4 COMPLETED SUCCESSFULLY] Customer RFM Mart is ready for modeling.")
    print("=" * 65)

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "소비자.csv")
    out_parquet = os.path.join("data", "processed", "customer_rfm_features.parquet")
    out_summary = os.path.join("data", "processed", "rfm_feature_summary.json")
    build_rfm_features(raw_path, out_parquet, out_summary)
