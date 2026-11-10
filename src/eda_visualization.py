"""
EDA Stage 1 Visualization: Key Diagnostic Plots
File: src/eda_visualization.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set matplotlib font for Korean support
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_eda_charts(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data for visualization: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 1. Transaction Amount Distribution (Log Scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    log_amt = np.log10(df["transaction_amt"])
    sns.histplot(log_amt, bins=40, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("거래 금액 분포 (Log10 Scale)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Log10(거래 금액, 원)", fontsize=11)
    ax.set_ylabel("거래 건수", fontsize=11)
    ax.axvline(log_amt.median(), color="red", linestyle="--", label=f"중앙값: 1,000만원 (10^7)")
    ax.axvline(np.log10(df["transaction_amt"].mean()), color="green", linestyle=":", label=f"평균: 5,443만원")
    ax.legend(fontsize=10)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_transaction_amt_dist.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart1_path}")

    # 2. Customer Transaction Frequency Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cust_counts = df["customer_id"].value_counts().value_counts().sort_index()
    bars = ax.bar(cust_counts.index, cust_counts.values, color="#ff7f0e", width=0.6)
    ax.set_title("고객별 거래 빈도(Frequency) 분포", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("고객당 누적 거래 횟수", fontsize=11)
    ax.set_ylabel("고객 수 (Log Scale)", fontsize=11)
    ax.set_yscale("log")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{yval:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(cust_counts.index)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_frequency_distribution.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart2_path}")

    # 3. Demographic Distributions (Age & Gender & Sector)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 3-1. Age
    age_order = ["20대", "30대", "40대", "50대", "60대", "70대", "70대 이상", "80대 이상"]
    age_counts = df["age"].value_counts().reindex([a for a in age_order if a in df["age"].values])
    axes[0].bar(age_counts.index, age_counts.values, color="#2ca02c")
    axes[0].set_title("연령대별 분포", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("건수")

    # 3-2. Gender
    gender_counts = df["gender"].value_counts()
    axes[1].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%", colors=["#3498db", "#e74c3c"], startangle=90)
    axes[1].set_title("성별 구성비", fontsize=12, fontweight="bold")

    # 3-3. Sector
    sector_counts = df["customer_type"].value_counts()
    axes[2].pie(sector_counts.values, labels=sector_counts.index, autopct="%1.1f%%", colors=["#9b59b6", "#f1c40f"], startangle=90)
    axes[2].set_title("업종(증권 vs 보험) 구성비", fontsize=12, fontweight="bold")

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_demographics_overview.png")
    fig.savefig(chart3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart3_path}")

    # 4. Sector-wise Amount Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="customer_type", y="transaction_amt", data=df, palette="Set2", ax=ax, showfliers=False)
    ax.set_title("업종별 거래 금액 비교 (이상치 제외 Boxplot)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("업종 구분", fontsize=11)
    ax.set_ylabel("거래 금액 (원)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*1e-4:,.0f}만원" if x < 1e8 else f"{x*1e-8:,.1f}억원"))
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_sector_amount_comparison.png")
    fig.savefig(chart4_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart4_path}")

    print("[SUCCESS] All 4 EDA charts successfully generated.")

if __name__ == "__main__":
    raw_csv = os.path.join("data", "raw", "소비자.csv")
    out_dir = os.path.join("reports", "figures")
    create_eda_charts(raw_csv, out_dir)
