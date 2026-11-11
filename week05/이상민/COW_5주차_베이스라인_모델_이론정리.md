# COW 5주차 | 머신러닝을 활용한 베이스라인 모델 구축

> **주차 주제:** 머신러닝 또는 딥러닝을 활용한 베이스라인 모델 구축  
> **프로젝트:** 고객 구매행동 데이터 기반 K-Means 군집화  
> **범위:** 베이스라인의 개념, 모델 선택, 입력 데이터, K-Means의 원리, 구현  
> **이번 문서에서 제외:** 학습 방법론 비교, 성능 평가 지표, 성능 향상 실험

---

## 1. 베이스라인 모델(Baseline Model)

베이스라인 모델은 프로젝트에서 **처음 구현하는 기준 모델**이다. 복잡한 개선 기법을 적용하기 전에 데이터 입력부터 모델 실행, 결과 생성까지 기본 흐름이 정상적으로 작동하는지 확인하는 데 사용한다.

우리 프로젝트에서는 정형 고객 데이터를 입력으로 받아 고객별 군집 번호를 생성하는 **K-Means 모델**을 베이스라인으로 설정했다.

### 기본 구축 흐름

```text
프로젝트 목적 설정
       ↓
알고리즘 선택
       ↓
입력 변수 준비
       ↓
필요한 전처리
       ↓
모델 생성 및 실행
       ↓
고객별 군집 번호 확인
```

## 2. 머신러닝과 딥러닝

- **머신러닝(Machine Learning):** 데이터의 특성이나 구조를 활용해 규칙과 패턴을 찾는 기술.
- **딥러닝(Deep Learning):** 여러 층의 인공신경망을 활용하는 머신러닝의 한 분야.

| 구분 | 머신러닝 | 딥러닝 |
|---|---|---|
| 대표 기법 | K-Means, 의사결정나무, 랜덤 포레스트 | CNN, RNN, Transformer |
| 주요 구조 | 알고리즘마다 다름 | 다층 신경망 |
| 이번 프로젝트 | **K-Means 사용** | 사용하지 않음 |

> K-Means는 **비지도 머신러닝 알고리즘**이며 딥러닝 모델은 아니다. 비지도학습은 고객 유형처럼 미리 주어진 정답 레이블 없이 데이터의 구조를 찾는 방식이다.

## 3. 군집화(Clustering)

군집화는 **서로 유사한 특성을 가진 관측값을 같은 그룹으로 묶는 작업**이다. 각 그룹을 *군집(Cluster)*이라고 한다.

예를 들어 구매금액과 구매빈도를 기준으로 고객을 표현하면 다음과 같은 유형을 생각할 수 있다.

| 예시 유형 | 구매금액 | 구매빈도 |
|---|---|---|
| A | 높음 | 높음 |
| B | 높음 | 낮음 |
| C | 낮음 | 높음 |
| D | 낮음 | 낮음 |

이 표는 **개념 설명용 가상 예시**이며 실제 데이터에서 발견한 군집은 아니다. K-Means의 군집은 사람이 미리 붙인 유형명이 아니라, 데이터의 거리 구조에 따라 생성된다.

## 4. K-Means의 기본 원리

K-Means는 관측값을 미리 지정한 **K개 군집**으로 나누는 알고리즘이다. 각 군집은 **중심점(Centroid)**을 가지며, 중심점은 해당 군집 데이터의 변수별 평균 위치다.

### 4.1 중심점(Centroid)

세 고객의 좌표가 아래와 같다고 가정하자.

| 고객 | 구매금액 | 구매빈도 |
|---|---:|---:|
| A | 20 | 2 |
| B | 30 | 4 |
| C | 40 | 6 |

세 고객이 같은 군집에 속한다면 중심점은

\[
\mu = \left(\frac{20+30+40}{3},\frac{2+4+6}{3}\right)=(30,4)
\]

이다.

### 4.2 군집 생성 과정

1. 군집 수 `K`를 지정한다.
2. `K`개의 초기 중심점을 정한다.
3. 각 데이터를 가장 가까운 중심점에 할당한다.
4. 각 군집에 속한 데이터의 평균으로 중심점을 갱신한다.
5. 할당과 갱신을 반복하여 종료 조건에 도달하면 군집 번호를 반환한다.

### 4.3 유클리드 거리(Euclidean Distance)

K-Means는 일반적으로 **유클리드 거리의 제곱**을 기준으로 가까운 중심점을 판단한다. 2차원에서 두 점 사이의 유클리드 거리는 다음과 같다.

\[
d(A,B)=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

예를 들어 `(20, 2)`와 `(30, 4)` 사이의 거리는

\[
\sqrt{(30-20)^2+(4-2)^2}=\sqrt{104}\approx 10.20
\]

이다. **거리 기반 알고리즘에서는 변수의 단위와 범위가 중요하다.**

## 5. 프로젝트 입력 변수(Feature)

우리 프로젝트의 초기 베이스라인은 고객 **3,900명**의 구매행동을 다음 세 변수로 표현했다.

| 변수 | 의미 | 주의할 점 |
|---|---|---|
| `Purchase Amount (USD)` | 현재 기록된 구매금액 | 누적 구매금액이 아님 |
| `Previous Purchases` | 이전 구매횟수 | 관측된 과거 구매횟수 |
| `Purchase Frequency Per Year` | 구매주기를 연간 횟수로 환산한 값 | 실제 연간 거래횟수가 아닌 **추정값** |

원본 `Frequency of Purchases`는 `Weekly`, `Monthly` 등 문자열 범주로 기록되어 있으므로 수치형 변수로 변환해 사용했다.

| 원래 구매주기 | 연간 환산값 |
|---|---:|
| Weekly | 약 52.14 |
| Bi-Weekly / Fortnightly | 약 26.07 |
| Monthly | 12 |
| Every 3 Months / Quarterly | 4 |
| Annually | 1 |

> **해석상의 주의:** 구매주기를 수치로 바꾼 값은 편의를 위한 환산값이며, 실제 고객별 연간 주문 이력을 복원한 값이 아니다.

## 6. 데이터 스케일링(Scaling)

K-Means는 거리 기반 알고리즘이므로 변수마다 값의 범위가 다르면 특정 변수가 거리 계산에 더 크게 반영될 수 있다. 이를 조정하기 위해 스케일링을 사용한다.

### StandardScaler

변수별 평균을 빼고 표준편차로 나누는 방식이다.

\[
z=\frac{x-\mu}{\sigma}
\]

초기 베이스라인에서는 **StandardScaler**를 사용했다.

### RobustScaler

변수별 중앙값을 빼고 사분위 범위(IQR)로 나누는 방식이다.

\[
x'=\frac{x-\operatorname{Median}(X)}{Q_3-Q_1}
\]

RobustScaler는 후속 실험에서 사용한 전처리 방식이다. 이번 문서에서는 **개념만 소개**하며, 스케일링 간 결과 비교는 다루지 않는다.

## 7. Python 구현 예시

아래 코드는 `shopping_behavior_updated.csv`가 현재 작업 폴더에 있다고 가정한 **독립 실행 예시**다. 기존 Jupyter Notebook에서 `df`를 이미 준비했다면 CSV를 읽는 부분은 생략할 수 있다.

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
df = pd.read_csv("shopping_behavior_updated.csv")

# 2. 범주형 구매주기를 연간 횟수로 환산
frequency_map = {
    "Weekly": 365 / 7,
    "Bi-Weekly": 365 / 14,
    "Fortnightly": 365 / 14,
    "Monthly": 12,
    "Every 3 Months": 4,
    "Quarterly": 4,
    "Annually": 1,
}

df["Purchase Frequency Per Year"] = (
    df["Frequency of Purchases"].map(frequency_map)
)

# 3. 입력 변수 선택
features = [
    "Purchase Amount (USD)",
    "Previous Purchases",
    "Purchase Frequency Per Year",
]
X = df[features].copy()

# 매핑되지 않은 구매주기나 결측값이 없는지 확인
if X.isna().any().any():
    raise ValueError("입력 변수에 결측값이 있습니다. 원본 데이터를 확인하세요.")

# 4. 변수 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. K-Means 베이스라인 모델 생성 및 실행
kmeans = KMeans(n_clusters=8, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 6. 생성된 군집 번호 확인
print(df[["Customer ID", "Cluster"]].head())
print(df["Cluster"].value_counts().sort_index())
```

### 주요 코드 요소

| 코드 | 역할 |
|---|---|
| `StandardScaler()` | 입력 변수 표준화 객체 생성 |
| `fit_transform(X)` | 입력 데이터에 스케일링 적용 |
| `KMeans(n_clusters=8)` | 8개 군집을 생성하는 모델 설정 |
| `random_state=42` | 무작위 초기화의 재현성을 위한 시드 설정 |
| `n_init=20` | 서로 다른 초기 중심점으로 20회 실행 |
| `fit_predict(X_scaled)` | 군집화 실행 후 각 고객의 군집 번호 반환 |

`Cluster` 값은 `0`부터 `7`까지의 **식별 번호**일 뿐이며, 숫자가 높다고 더 우수한 고객을 뜻하지 않는다.

## 8. 이번 주차 프로젝트 적용 내용

- **데이터:** 고객 3,900명의 구매행동 데이터
- **모델:** K-Means
- **초기 입력 변수:** 구매금액, 이전 구매횟수, 연간 환산 구매빈도
- **전처리:** StandardScaler
- **초기 군집 수:** K=8
- **구현 결과물:** 고객별 군집 번호(`Cluster`)가 추가된 데이터프레임

이번 주차의 핵심은 **기본 모델을 구성하고 고객별 군집 할당 결과를 생성하는 것**이다. 군집 품질의 정량적 평가, 모델 간 비교, 개선 실험은 별도 주차에서 다룬다.

---

## 핵심 용어 요약

| 용어 | 의미 |
|---|---|
| Baseline Model | 프로젝트의 초기 기준 모델 |
| Machine Learning | 데이터에서 규칙이나 구조를 찾는 기술 |
| Deep Learning | 다층 인공신경망을 사용하는 머신러닝 분야 |
| Clustering | 유사한 데이터를 그룹으로 묶는 작업 |
| K-Means | 중심점을 기준으로 K개 군집을 형성하는 알고리즘 |
| Centroid | 군집 내 데이터의 평균 위치 |
| Euclidean Distance | 두 점 사이의 직선거리 |
| Feature | 모델에 입력하는 변수 |
| Scaling | 변수의 수치 범위를 조정하는 전처리 |

> **한 줄 정리:** COW 5주차에는 고객 구매행동의 세 변수를 준비하고 표준화한 뒤, K-Means로 고객별 군집 번호를 생성하는 베이스라인 모델을 구축했다.
