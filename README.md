# Finance Customer Segmentation & RFM Intelligence

> **AI Hub 금융상품·서비스 및 소비자 특성 데이터(SN 71938)** 기반 고객 거래 행동 분석, RFM 피처 엔지니어링, K-Means 군집화 및 비즈니스 의사결정 지원 대화형 웹 대시보드

---

## 1. 프로젝트 개요 (Executive Summary)

본 프로젝트는 금융 소비자의 실제 대규모 금융 거래 내역(385,019건, 335,506명)을 기반으로 **고객 행동(RFM)과 인구통계학적·재무 성향 프로필을 결합한 하이브리드 고객 세분화(Hybrid Customer Segmentation)** 파이프라인을 구축한 프로젝트입니다.

데이터셋 초기 정밀 점검(Feasibility Assessment)부터 탐색적 데이터 분석(EDA), 다차원 피처 엔지니어링, K-Means 머신러닝 군집화, 4대 비즈니스 페르소나 도출, 그리고 임직원 의사결정용 인터랙티브 웹 대시보드 구축까지 엔드투엔드로 완결되었습니다.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 핵심 분석 요약 지표                                    │
├────────────────────┬───────────────────────────────────────────────────────────────────┤
│ 총 분석 고객 수    │ 335,506명 (고유 식별자 customer_id 보유)                           │
│ 총 분석 거래 건수  │ 385,019건 (증권 79.9%, 보험 20.1%)                                │
│ 평균 누적 거래액   │ 6,246만 원 (중앙값 1,000만 원, 최대 160억 원)                      │
│ 최적 군집 수 ($K$) │ K=4 (Inertia 굴곡점 및 실루엣 계수 0.3476, 비즈니스 분리도 최상) │
│ 프라이버시 보호    │ 100% 집계(Aggregated) 데이터만 대시보드 및 리포트에 공개           │
└────────────────────┴───────────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터셋 정밀 검증 및 적합성 판정 (Dataset Feasibility)

* **데이터 출처:** AI Hub 「금융상품·서비스 및 소비자 특성 데이터」(한국지능정보사회진흥원, SN: 71938)
* **초기 점검과 실제 패키지 발견:**
  * AI Hub 온라인 소개 페이지에서는 LLM CoT(Chain-of-Thought) 질의응답 샘플(1만 건) 위주로 기술되어 있었으나, 실제 로컬 다운로드 패키지 내부를 정밀 스캔한 결과 **실제 거래 원천 데이터(`소비자.csv`, 385,019행, 22개 컬럼)**가 완벽히 탑재되어 있음을 확인하였습니다.
  * 이에 따라 **[Category A: 거래 기반 고객 세분화 및 RFM 분석 완벽 지원]**으로 적합성을 최종 판정하였습니다.

### 22개 주요 컬럼 구조
1. **식별 및 거래:** `customer_id` (해시 고유값), `customer_seq` (거래 회차 1~7), `transaction_type`, `transaction_amt`, `transaction_start_year`, `transaction_last_year`
2. **금융 상품:** `customer_type` (증권 307,714건, 보험 77,305건), `product_name`, `product_amt`, `product_period`
3. **고객 프로필:** `age`, `gender`, `region`, `income_bracket`, `occupation_group`, `risk_grade`, `credit_rating`, `marital_status`, `investment_propensity`

---

## 3. 분석 파이프라인 아키텍처

```
[1. 원천 거래 데이터] ──> [2. 탐색적 데이터 분석(EDA)] ──> [3. 고객 단위 RFM 집계]
 - 소비자.csv (385k 행)      - 결측치/이상치 검증          - 335k명 1인 1행 변환
 - 무결성 검증 완료          - 거래액 및 연령/성별 분석     - Recency, Frequency, Monetary

                                                                      │
                                                                      ▼
[6. 웹 대시보드 개발] ◄── [5. 비즈니스 페르소나 정의] ◄── [4. 피처 스케일링 & K-Means]
 - dashboard/index.html      - 4대 타겟 CRM 전략 수립       - log1p 변환 (왜도 해소)
 - Chart.js & Tailwind       - LTV 극대화 & 윈백 플랜       - StandardScaler (K=4 모델링)
```

---

## 4. 고객 단위 RFM 피처 엔지니어링

원천 데이터(거래 1건당 1행)를 **고객 1인당 1행(335,506행)**의 고속 Parquet 마트(`customer_rfm_features.parquet`)로 변환하였습니다.

| RFM 지표 | 산출 로직 | 기초 통계치 (평균 / 중앙값) | 비즈니스 의미 |
| :--- | :--- | :--- | :--- |
| **Recency (최근성)** | `2025 - max(transaction_last_year)` | 평균 **0.54년** / 중앙값 **0.0년** | 최근 거래 경과 연수 (89.9%가 2025년 거래) |
| **Frequency (빈도)** | `count(customer_seq)` | 평균 **1.15회** / 중앙값 **1.0회** | 누적 거래 건수 (13.16%가 2회 이상 다회 거래) |
| **Monetary (금액)** | `sum(transaction_amt)` | 평균 **6,246만 원** / 중앙값 **1,000만 원** | 고객당 총 누적 거래 기여 금액 (최대 160억 원) |
| **Tenure (관계연수)** | `2025 - min(transaction_start_year)` | 평균 **3.82년** / 최대 **37년** | 최초 거래 시점부터의 누적 거래 관계 연수 |

### 전처리 및 스케일링
* **로그 변환 (`log1p`):** 거래 금액(Monetary)과 빈도(Frequency)의 극단적인 우측 왜도(Right Skewness)를 정규분포 형태로 안정화.
* **표준화 (`StandardScaler`):** 유클리디안 거리 기반 K-Means 클러스터링을 위해 평균 0, 분산 1로 스케일링 적용.

---

## 5. K-Means 군집화 및 4대 비즈니스 페르소나

다차원 지표(`recency_log_std`, `frequency_log_std`, `monetary_log_std`, `tenure_log_std`, `age_ordinal_std`, `income_ordinal_std`)를 기반으로 모델링한 결과, **$K=4$**에서 가장 명확하고 실효성 있는 고객 세분화가 도출되었습니다.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              4대 고객 세그먼트 프로필 요약                             │
├──────────┬────────────────────────────┬─────────┬────────┬────────────┬────────────────┤
│ 군집 번호 │ 세그먼트 페르소나 명       │ 고객 수 │ 비율(%)│ 평균거래액 │ 다회거래율(%)  │
├──────────┼────────────────────────────┼─────────┼────────┼────────────┼────────────────┤
│ Cluster 0│ VIP 다빈도 장기 충성 고객  │  41,815 │  12.5% │ 1억 5,577만│         100.0% │
│ Cluster 1│ 신규 유입 소액 보험 고객  │  77,842 │  23.2% │     247만 │           0.9% │
│ Cluster 2│ 고액 자산가 이탈 징후 고객 │  26,914 │   8.0% │ 1억 5,018만│           6.0% │
│ Cluster 3│ 증권 단발 중위 자산 고객  │ 188,935 │  56.3% │   5,404만 │           0.0% │
└──────────┴────────────────────────────┴─────────┴────────┴────────────┴────────────────┘
```

### [군집별 상세 특성 및 맞춤형 CRM 액션 플랜]

1. **Cluster 0: VIP 다빈도 장기 충성 고객 (12.5%, 41,815명)**
   * **특징:** 100% 다회 거래 고객, 평균 거래액 **1억 5,577만 원**, 평균 관계 연수 **14.2년**, 증권 비중 96.1%.
   * **CRM 전략:** 1:1 전담 PB 매칭, 사모펀드·글로벌 배당 랩어카운트 우선 배정, 고액 신탁 크로스셀링으로 락인(Lock-in) 및 LTV 극대화.
2. **Cluster 1: 신규 유입 소액 보험 고객 (23.2%, 77,842명)**
   * **특징:** 평균 관계 연수 **0.87년 (1년 미만 신규)**, 평균 거래액 **247만 원**, 보험 비중 **92.05%**.
   * **CRM 전략:** 모바일 앱 헬스케어 리포트 및 보장 분석 무료 제공, 첫 거래 감사 리워드, 소액 적립식 펀드/ISA 계좌 개설 프로모션.
3. **Cluster 2: 고액 자산가 이탈 징후 고객 (8.0%, 26,914명)**
   * **특징:** 최근 거래 미발생 경과 **평균 6.43년 (장기 휴면)**, 평균 거래액 **1억 5,018만 원**, **전 군집 중 최고 소득 계층**.
   * **CRM 전략:** 휴면 자산 점검 리포트 발송, '자산 리밸런싱 세미나' 초대, 단기 고금리 특판 채권 우선 배정 및 수수료 우대 혜택으로 긴급 윈백(Win-back) 추진.
4. **Cluster 3: 증권 단발 중위 자산 고객 (56.3%, 188,935명)**
   * **특징:** 당사 최다 볼륨 고객군, 평균 거래액 **5,404만 원**, 평균 관계 연수 **11.6년**, 증권 비중 **99.6%**.
   * **CRM 전략:** 단발성 거래를 다회 거래로 전환하기 위한 '적립식 자동이체 챌린지', 연금저축·IRP 세액공제 특화 상품 추천 캠페인 운영.

---

## 6. 인터랙티브 웹 대시보드 (`dashboard/index.html`)

임원진 및 마케팅 실무진이 웹 브라우저를 통해 실시간으로 분석 결과를 탐색할 수 있는 정적 웹 애플리케이션입니다.

* **기술 스택:** HTML5, Tailwind CSS, Chart.js, FontAwesome
* **보안:** 개인정보가 전혀 없는 집계 JSON(`dashboard/data/dashboard_summary.json`)만을 비동기 연동.
* **실행 방법:**
  1. `dashboard/index.html` 파일을 더블 클릭하여 웹 브라우저에서 바로 열거나,
  2. 프로젝트 루트에서 다음 명령어를 실행하여 로컬 서버로 접속:
     ```powershell
     python -m http.server 8000
     ```
     브라우저 주소: `http://localhost:8000/dashboard/index.html`

---

## 7. 저장소 디렉터리 구조

```
finance-customer-segmentation/
├── data/
│   ├── raw/                      # 원본 데이터 (Git 미추적)
│   │   └── 소비자.csv            # 385,019건 원천 거래 내역 (101.3 MB)
│   └── processed/                # 전처리 마트 데이터 (Git 미추적)
│       ├── customer_rfm_features.parquet  # 335k명 RFM 마트 (25.3 MB)
│       ├── customer_clusters.parquet      # 군집 라벨 포함 마트 (26.5 MB)
│       ├── eda_summary.json               # EDA 정량 통계 요약
│       └── rfm_feature_summary.json       # RFM 마트 통계 요약
├── src/                          # 파이썬 실행 모듈
│   ├── eda_stage1.py             # 탐색적 데이터 분석 스크립트
│   ├── eda_visualization.py      # 진단 시각화 차트 생성 스크립트
│   ├── feature_engineering.py   # 고객 단위 RFM 피처 엔지니어링 모듈
│   └── train_kmeans.py           # K-Means 모델링 & 페르소나 프로파일링 모듈
├── reports/
│   └── figures/                  # 시각화 산출물 (200 DPI)
│       ├── 01_transaction_amt_dist.png      # 거래 금액 분포
│       ├── 02_frequency_distribution.png    # 거래 빈도 분포
│       ├── 03_demographics_overview.png     # 연령/성별/업종 구성비
│       ├── 04_sector_amount_comparison.png  # 업종별 거래액 박스플롯
│       ├── 05_elbow_silhouette.png          # 엘보우 & 실루엣 점수 곡선
│       ├── 06_pca_clusters.png              # 2D PCA 차원축소 군집 투영
│       └── 07_cluster_profiles.png          # 군집별 핵심 지표 비교
├── dashboard/                    # 인터랙티브 웹 대시보드
│   ├── index.html                # 단일 파일 대시보드 웹앱
│   └── data/
│       └── dashboard_summary.json # 공개용 안전 집계 데이터 (PII 0건)
├── .gitignore                    # 원본 데이터 및 가상환경 격리
└── README.md                     # 프로젝트 종합 총괄 보고서
```

---

## 8. 데이터 거버넌스 및 재현 가이드

1. **가상환경 활성화 및 패키지 설치:**
   ```powershell
   uv venv
   .venv\Scripts\activate
   uv pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow
   ```
2. **파이프라인 순차 실행:**
   ```powershell
   python src/eda_stage1.py
   python src/eda_visualization.py
   python src/feature_engineering.py
   python src/train_kmeans.py
   ```
3. **데이터 보호 준수:**
   * AI-Hub 이용 정책에 따라 원본 데이터(`data/raw/`) 및 개인별 마트(`data/processed/`)는 `.gitignore`를 통해 Git 추적에서 제외되어 외부 노출이 철저히 차단됩니다.
