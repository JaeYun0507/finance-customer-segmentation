"""
K-Means Clustering Pipeline for Financial Customer Segmentation
File: src/train_kmeans.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Korean font setup for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_kmeans_pipeline(parquet_path: str, output_parquet_path: str, figures_dir: str, dashboard_json_path: str):
    print("=" * 65)
    print(" [STEP 6] K-Means Clustering & Customer Segmentation Pipeline")
    print("=" * 65)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Processed feature file not found: {parquet_path}")

    print(f"Loading feature mart: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} customers with {len(df.columns)} columns.")

    # 1. Feature Selection for Clustering
    feature_cols = [
        "recency_log_std",
        "frequency_log_std",
        "monetary_log_std",
        "tenure_log_std",
        "age_ordinal_std",
        "income_ordinal_std"
    ]
    print(f"\n1. Clustering features selected ({len(feature_cols)}): {feature_cols}")
    X = df[feature_cols].values

    # 2. Optimal K Exploration (Elbow Method & Silhouette Score)
    print("\n2. Evaluating cluster quality across K = 2 to 6...")
    k_range = list(range(2, 7))
    inertias = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    # Sampling 30,000 records for fast and reliable silhouette score calculation
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=min(30000, len(df)), replace=False)
    X_sample = X[sample_indices]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X)
        inertias.append(km.inertia_)
        
        # Evaluation on sample
        km_sample = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sample)
        sil = silhouette_score(X_sample, km_sample.labels_)
        ch = calinski_harabasz_score(X_sample, km_sample.labels_)
        db = davies_bouldin_score(X_sample, km_sample.labels_)
        
        silhouette_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        print(f" - K={k}: Inertia={km.inertia_:,.1f}, Silhouette={sil:.4f}, Calinski-Harabasz={ch:,.1f}, Davies-Bouldin={db:.4f}")

    # Plot 1: Elbow Curve & Silhouette Score
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = '#1f77b4'
    ax1.set_xlabel('클러스터 수 (K)', fontsize=12, labelpad=8)
    ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)
    ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2, label='Inertia (Elbow)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('실루엣 점수 (Silhouette Score)', color=color, fontsize=12)
    ax2.plot(k_range, silhouette_scores, marker='s', color=color, linewidth=2, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('최적 군집 수(K) 탐색: Elbow Method & Silhouette Score', fontsize=14, fontweight='bold', pad=14)
    fig.tight_layout()
    chart1_path = os.path.join(figures_dir, "05_elbow_silhouette.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart1_path}")

    # 3. Final Model Selection (K=4: Strong balance of distinct business personas & cluster stability)
    OPTIMAL_K = 4
    print(f"\n3. Training Final K-Means Model with K={OPTIMAL_K}...")
    final_km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=15, max_iter=300)
    df["cluster"] = final_km.fit_predict(X)

    # 4. Cluster Persona Profiling
    print("\n4. Characterizing Cluster Personas...")
    cluster_profiles = []
    
    # We will identify personas based on their real RFM and demographic metrics
    for c in range(OPTIMAL_K):
        c_df = df[df["cluster"] == c]
        count = len(c_df)
        pct = (count / len(df)) * 100
        
        r_mean = c_df["recency_years"].mean()
        f_mean = c_df["frequency"].mean()
        m_mean = c_df["monetary_total"].mean()
        m_median = c_df["monetary_total"].median()
        t_mean = c_df["tenure_years"].mean()
        age_mean = c_df["age_ordinal"].mean()
        inc_mean = c_df["income_ordinal"].mean()
        sec_pct = c_df["sector_is_securities"].mean() * 100
        repeat_pct = c_df["is_repeat_customer"].mean() * 100
        
        cluster_profiles.append({
            "cluster_id": c,
            "count": int(count),
            "percentage": round(float(pct), 2),
            "recency_years_mean": round(float(r_mean), 2),
            "frequency_mean": round(float(f_mean), 2),
            "repeat_customer_pct": round(float(repeat_pct), 2),
            "monetary_mean_krw": round(float(m_mean), 0),
            "monetary_median_krw": round(float(m_median), 0),
            "tenure_years_mean": round(float(t_mean), 2),
            "age_ordinal_mean": round(float(age_mean), 2),
            "income_ordinal_mean": round(float(inc_mean), 2),
            "securities_pct": round(float(sec_pct), 2)
        })

    # Sort profiles to assign standardized business names
    # Priority for naming: High Monetary & High Freq -> VIP, Long Tenure -> Long-term, High Recency -> Dormant, etc.
    profiles_df = pd.DataFrame(cluster_profiles)
    print("\nCluster Baseline Metrics Table:")
    print(profiles_df[["cluster_id", "percentage", "recency_years_mean", "frequency_mean", "monetary_mean_krw", "tenure_years_mean"]].to_string())

    # Map meaningful Persona Labels based on characteristics
    # Assign names dynamically based on stats
    name_mapping = {}
    for p in cluster_profiles:
        cid = p["cluster_id"]
        if p["recency_years_mean"] > 3.0:
            name_mapping[cid] = "이탈 징후 장기 정체 고객 (Dormant)"
        elif p["frequency_mean"] > 1.3 or p["repeat_customer_pct"] > 30:
            name_mapping[cid] = "VIP 다빈도 충성 고객 (VIP Loyal)"
        elif p["monetary_mean_krw"] > 70000000:
            name_mapping[cid] = "고액 자산가 우량 고객 (High-Net-Worth)"
        else:
            name_mapping[cid] = "신규 및 일반 실속 고객 (Mass Standard)"

    # Update names
    for p in cluster_profiles:
        p["persona_name"] = name_mapping[p["cluster_id"]]
        print(f" - Cluster {p['cluster_id']}: [{p['persona_name']}] - {p['count']:,}명 ({p['percentage']}%), 평균금액: {p['monetary_mean_krw']:,.0f}원")

    df["cluster_name"] = df["cluster"].map(name_mapping)

    # 5. Visualizations
    print("\n5. Generating Cluster Visualization Charts...")

    # Plot 2: PCA 2D Projection
    pca = PCA(n_components=2, random_state=42)
    pca_sample = pca.fit_transform(X_sample)
    sample_clusters = df.loc[sample_indices, "cluster"].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", OPTIMAL_K)
    for c in range(OPTIMAL_K):
        mask = sample_clusters == c
        ax.scatter(pca_sample[mask, 0], pca_sample[mask, 1], label=f"C{c}: {name_mapping[c]}", alpha=0.5, s=20, color=palette[c])
    
    ax.set_title(f"K-Means 고객 세분화 2D PCA 투영 (설명 분산비: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(f"PCA 컴포넌트 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PCA 컴포넌트 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "06_pca_clusters.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart2_path}")

    # Plot 3: Cluster Radar / Profile Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 3-1: Monetary Comparison
    m_means = [p["monetary_mean_krw"] / 1e4 for p in cluster_profiles]
    axes[0].bar([f"C{p['cluster_id']}\n({p['persona_name'].split()[0]})" for p in cluster_profiles], m_means, color='#2ca02c')
    axes[0].set_title("군집별 평균 거래 금액 (만 원)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("평균 금액 (만 원)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 3-2: Frequency & Repeat Customer Pct
    f_means = [p["repeat_customer_pct"] for p in cluster_profiles]
    axes[1].bar([f"C{p['cluster_id']}\n({p['persona_name'].split()[0]})" for p in cluster_profiles], f_means, color='#ff7f0e')
    axes[1].set_title("군집별 반복 거래 고객 비율 (%)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("반복 거래 비율 (%)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # 3-3: Tenure (Relationship Years)
    t_means = [p["tenure_years_mean"] for p in cluster_profiles]
    axes[2].bar([f"C{p['cluster_id']}\n({p['persona_name'].split()[0]})" for p in cluster_profiles], t_means, color='#1f77b4')
    axes[2].set_title("군집별 평균 관계 유지 연수 (년)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("평균 연수 (년)")
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "07_cluster_profiles.png")
    fig.savefig(chart3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart3_path}")

    # 6. Save Clustered Dataset & Dashboard Summary
    print("\n6. Saving clustered data and public dashboard summary...")
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    print(f" - Clustered dataset saved to: {output_parquet_path}")

    # Public dashboard summary JSON (Aggregated results only - strictly no customer-level PII)
    dashboard_payload = {
        "k_evaluation": {
            "k_values": k_range,
            "inertias": [float(i) for i in inertias],
            "silhouette_scores": [round(float(s), 4) for s in silhouette_scores],
            "calinski_harabasz": [round(float(c), 1) for c in ch_scores],
            "davies_bouldin": [round(float(d), 4) for d in db_scores]
        },
        "optimal_k": OPTIMAL_K,
        "pca_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "clusters": cluster_profiles,
        "metadata": {
            "total_customers": len(df),
            "features_used": feature_cols,
            "reference_year": 2025
        }
    }

    os.makedirs(os.path.dirname(dashboard_json_path), exist_ok=True)
    with open(dashboard_json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, ensure_ascii=False, indent=2)
    print(f" - Dashboard summary JSON saved to: {dashboard_json_path}")
    print("=" * 65)
    print(" [STEP 6 COMPLETED SUCCESSFULLY] K-Means Clustering & Profiling Done.")
    print("=" * 65)

if __name__ == "__main__":
    in_parquet = os.path.join("data", "processed", "customer_rfm_features.parquet")
    out_parquet = os.path.join("data", "processed", "customer_clusters.parquet")
    fig_dir = os.path.join("reports", "figures")
    dash_json = os.path.join("dashboard", "data", "dashboard_summary.json")
    run_kmeans_pipeline(in_parquet, out_parquet, fig_dir, dash_json)
