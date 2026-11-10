"""
EDA Stage 1: Financial Customer Dataset Baseline Inspection
File: src/eda_stage1.py
"""

import os
import json
import pandas as pd
import numpy as np

def run_eda(file_path: str, output_json_path: str):
    print("=" * 60)
    print(" [EDA STEP 1] Financial Customer Dataset Baseline Analysis")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    
    # 1. Dataset Dimensions
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\n1. Basic Dimensions:")
    print(f" - Rows: {n_rows:,}")
    print(f" - Columns: {n_cols}")
    print(f" - Memory Usage: {mem_mb:.2f} MB")
    
    # 2. Missing Values Analysis
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / n_rows) * 100
    missing_df = pd.DataFrame({"Missing_Count": missing_series, "Missing_Pct": missing_pct})
    missing_report = missing_df[missing_df["Missing_Count"] > 0]
    print(f"\n2. Missing Value Analysis:")
    if len(missing_report) == 0:
        print(" - No explicit null (NaN) values detected across all columns.")
    else:
        print(missing_report.to_string())
        
    # Check for '-' or placeholders in object columns
    placeholder_counts = {}
    for col in df.select_dtypes(include="object").columns:
        dash_count = (df[col] == "-").sum()
        if dash_count > 0:
            placeholder_counts[col] = int(dash_count)
    if placeholder_counts:
        print(f" - Placeholder '-' counts detected: {placeholder_counts}")

    # 3. Customer Identification & Transaction Frequency
    unique_customers = int(df["customer_id"].nunique())
    cust_tx_counts = df["customer_id"].value_counts()
    repeat_customers = int((cust_tx_counts > 1).sum())
    single_customers = int((cust_tx_counts == 1).sum())
    max_tx = int(cust_tx_counts.max())
    
    freq_dist = cust_tx_counts.value_counts().sort_index().to_dict()
    print(f"\n3. Customer Identification & Frequency Analysis:")
    print(f" - Unique Customers: {unique_customers:,}")
    print(f" - Single-transaction Customers: {single_customers:,} ({single_customers/unique_customers*100:.1f}%)")
    print(f" - Repeat-transaction Customers: {repeat_customers:,} ({repeat_customers/unique_customers*100:.1f}%)")
    print(f" - Max transactions per customer: {max_tx}")
    print(f" - Transaction frequency distribution (Tx count: Customer count): {freq_dist}")
    
    # 4. Sector & Transaction Types
    sector_dist = df["customer_type"].value_counts().to_dict()
    tx_type_dist = df["transaction_type"].value_counts().to_dict()
    print(f"\n4. Sector & Transaction Type:")
    print(f" - Customer Sector: {sector_dist}")
    print(f" - Transaction Type: {tx_type_dist}")
    
    # 5. Financial Figures: Transaction Amount
    amt_col = "transaction_amt"
    amt_desc = df[amt_col].describe().to_dict()
    zero_amt = int((df[amt_col] == 0).sum())
    negative_amt = int((df[amt_col] < 0).sum())
    print(f"\n5. Transaction Amount Summary (KRW):")
    print(f" - Min: {amt_desc['min']:,.0f}")
    print(f" - 25%: {amt_desc['25%']:,.0f}")
    print(f" - Median (50%): {amt_desc['50%']:,.0f}")
    print(f" - Mean: {amt_desc['mean']:,.0f}")
    print(f" - 75%: {amt_desc['75%']:,.0f}")
    print(f" - Max: {amt_desc['max']:,.0f}")
    print(f" - Zero count: {zero_amt}, Negative count: {negative_amt}")
    
    # 6. Timeline (Start Year & Last Year)
    start_yr_dist = df["transaction_start_year"].value_counts().sort_index().to_dict()
    last_yr_dist = df["transaction_last_year"].value_counts().sort_index().to_dict()
    print(f"\n6. Transaction Timeline:")
    print(f" - Start Year Range: {min(start_yr_dist.keys())} ~ {max(start_yr_dist.keys())}")
    print(f" - Last Year Range: {min(last_yr_dist.keys())} ~ {max(last_yr_dist.keys())}")
    print(f" - Recent Transactions (2025): {last_yr_dist.get(2025, 0):,} ({last_yr_dist.get(2025, 0)/n_rows*100:.1f}%)")

    # 7. Customer Demographic Breakdown
    age_dist = df["age"].value_counts().to_dict()
    gender_dist = df["gender"].value_counts().to_dict()
    income_dist = df["income_bracket"].value_counts().to_dict()
    credit_dist = df["credit_rating"].value_counts().to_dict()
    risk_dist = df["risk_grade"].value_counts().to_dict()
    
    print(f"\n7. Demographic & Credit Profiles:")
    print(f" - Age distribution: {age_dist}")
    print(f" - Gender distribution: {gender_dist}")
    print(f" - Income bracket distribution: {income_dist}")
    print(f" - Credit rating distribution: {credit_dist}")
    print(f" - Risk grade distribution: {risk_dist}")

    # Compile Structured Output for downstream tasks
    summary_output = {
        "dataset_dimensions": {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "memory_mb": round(mem_mb, 2)
        },
        "customer_metrics": {
            "unique_customers": unique_customers,
            "single_customers": single_customers,
            "repeat_customers": repeat_customers,
            "max_transactions": max_tx,
            "frequency_distribution": {str(k): int(v) for k, v in freq_dist.items()}
        },
        "financial_metrics": {
            "transaction_amt_stats": {k: float(v) for k, v in amt_desc.items()},
            "zero_amt_count": zero_amt,
            "negative_amt_count": negative_amt
        },
        "sector_distribution": {str(k): int(v) for k, v in sector_dist.items()},
        "timeline": {
            "start_year_min": int(min(start_yr_dist.keys())),
            "start_year_max": int(max(start_yr_dist.keys())),
            "last_year_min": int(min(last_yr_dist.keys())),
            "last_year_max": int(max(last_yr_dist.keys())),
            "recent_2025_count": int(last_yr_dist.get(2025, 0))
        },
        "demographics": {
            "age": {str(k): int(v) for k, v in age_dist.items()},
            "gender": {str(k): int(v) for k, v in gender_dist.items()},
            "income": {str(k): int(v) for k, v in income_dist.items()},
            "credit": {str(k): int(v) for k, v in credit_dist.items()},
            "risk_grade": {str(k): int(v) for k, v in risk_dist.items()}
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, ensure_ascii=False, indent=2)
    print(f"\n[SUCCESS] EDA Stage 1 summary saved to: {output_json_path}")
    print("=" * 60)

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "소비자.csv")
    out_path = os.path.join("data", "processed", "eda_summary.json")
    run_eda(raw_path, out_path)
