# NEAR-ZERO X — Counterfactual Predictive Safety OS

AI 기반 산업용 이동로봇 / 카트의 사고 예방 시스템.
"사고가 나기 **전에**, 지금 취할 수 있는 상반된 행동(counterfactual)을 **미리 시뮬레이션**해서
가장 위험을 줄이는 행동을 **추천**하고, 그 행동이 실제로 사고를 막았는지 **검증**하는
**PERCEIVE → PREDICT → PREVENT → VERIFY** 4단계 폐쇄루프 시스템입니다.

> 안전 판단은 AI(LLM)가 하지 않습니다. 모든 안전/위험 판단은 기하·확률·물리 기반의
> 결정적(deterministic) 파이프라인만 사용하며, 자연어 안전 규칙은 JSON DSL로 컴파일됩니다.

---

## 신속 실행 (Quick Start)

```powershell
# 1) 환경 설치 (CUDA GPU 기준, ~5분)
.\scripts\setup.ps1

# 2) 단위·통합 테스트 (38개)
.\scripts\run_tests.ps1

# 3) 평가 스위트 (정책 비교 + 성능 벤치마크 + ablation)
.\scripts\run_eval.ps1

# 4) 웹 대시보드 + 시뮬레이션 데모 (브라우저: http://localhost:5000)
.\scripts\run.ps1

# 5) 헤드리스 데모 (위험 추이 + 카운터팩추얼 표 + PREVENTED 검증 출력)
.\scripts\run_demo.ps1
```

수동 실행:

```powershell
.\.venv\Scripts\python.exe -m nearzero.demo                    # 데모 대시보드
.\.venv\Scripts\python.exe -m nearzero.demo --headless          # 본문 트랜스크립트 출력
.\.venv\Scripts\python.exe -m nearzero.demo --scenario crossing --headless --steps 60
.\.venv\Scripts\python.exe -m nearzero.eval.all                 # 전체 평가
```

---

## 시스템 아키텍처 (요약)

| 단계 | 모듈 | 역할 |
|---|---|---|
| PERCEIVE | `nearzero/vision/*`, `core/tracks.py` | YOLOv8n 감지 → 지면(Homography) 투영 → Kalman 추적 |
| PREDICT | `prediction/forecast.py` | 상호작용 인지 앙상블 궤적 예측 (불확실성 샘플 포함) |
| PREVENT | `risk/*`, `planner/*` | 확률 기반 위험 점수 + **counterfactual** 탐색 (행동별 resulting risk) |
| VERIFY | `verify/monitor.py` | PREDICTED→RECOMMENDED→**PREVENTED** 이벤트 상태 기계 |
| 안전 규칙 | `rules/*` | 자연어 규칙 → JSON DSL → 결정적 평가 (LLM 미사용) |

자세한 내용: [ARCHITECTURE.md](docs/ARCHITECTURE.md), [RISK_MODEL.md](docs/RISK_MODEL.md)

---

## 측정된 성능 (Measured, 가짜 값 없음)

### 정책 비교 (정합 시나리오 8종, recall/lead time)

| 정책 | recall | 평균 선행시간(lead) | 거짓 경보 |
|---|---|---|---|
| A: distance_threshold | 1.00 | 0.50 s | 1 |
| B: ttc_threshold | 1.00 | 2.62 s | 2 |
| C: trajectory_prediction | 1.00 | 2.70 s | 2 |
| **D: NEAR-ZERO X** | **1.00** | **1.52 s** | **1** |

### NEAR-ZERO X 세부 (플래그십 데모 `demo_avoided`)

- 위험 점수 0 → **49** (t=1.9 s, 충돌확률 72%) → `VEHICLE_RIGHT` 추천·실행 → 최저 **6** → 0
- 검증 이벤트: **`PREVENTED` 1건**
- 카운터팩추얼 최고안: VEHICLE_RIGHT (resulting risk 1) vs NO_ACTION 49

### 성능 벤치마크 (RTX 5070, GPU)

- 감지(YOLOv8n, CUDA): **48.7 ms/프레임**
- 코어 파이프라인(예측→위험→계획→검증): **181.6 ms/프레임**

### Ablation (NEAR-ZERO X 구성요소 유무)

| 변형 | recall | lead(s) | 경보 | 개입성공 | 평균위험감소 |
|---|---|---|---|---|---|
| full | 1.00 | 1.43 | 0 | 1.00 | 44.8 |
| 불확실성 제거 | 1.00 | 2.40 | 0 | 1.00 | 54.1 |
| counterfactual 제거 | 1.00 | 1.43 | 0 | NULL | NULL |
| 검증 제거 | 1.00 | 1.43 | 0 | 1.00 | 44.8 |

전체 표: [EVALUATION.md](docs/EVALUATION.md) · 원형 데이터: `results/comparison.csv`, `results/metrics.json`

---

## 디렉터리 구조

```
nearzero/
  app/          # 안전 파이프라인 조립, 상태 직렬화
  core/         # 기하(Homography), Kalman, 추적
  events/       # SQLite 이벤트 저장소 (thread-safe)
  planner/      # counterfactual 계획 + 개입 물리
  prediction/   # 앙상블 궤적 예측
  risk/         # 충돌확률·TTC·PMD·점수화·구역
  rules/        # 안전 규칙 DSL (자연어 파서 + 검증)
  server/       # Flask API, 시뮬레이션/카메라 러너
  sim/          # 세계 시뮬레이션 + 8개 시나리오
  verify/       # PREVENTED 검증 상태 기계
  vision/       # YOLOv8 감지, 어노테이션, 프라이버시
  demo/         # 데모 런처 (대시보드/헤드리스)
nearzero/eval/  # 정책 비교, metrics, benchmark, ablation
frontend/static/  # 웹 대시보드 (poller + MJPEG 영상)
scripts/        # PowerShell 실행 스크립트
docs/           # 문서 모음
tests/          # 단위·통합 테스트 (pytest, 38개)
results/        # 계측 산출물 (비교표, 보고서, plots)
configs/        # 규칙/구역 JSON + YAML 설정
```

---

## 법적/윤리 준수

- 안전 판단에 LLM 사용 금지 — 규칙 DSL은 결정적 평가 (`rules/engine.py`)
- CCTV 프라이버시 보호 — 즉시 얼굴 모자이크 (`vision/privacy.py`)
- 예방 목적의 경고/제한 기능만 구현, 자율 급제동 등은 시뮬레이션 한정