# RISK MODEL — 수식과 근거

위험 점수는 **모델 기반 추정치(model-based estimate)** 입니다.
Bayes 사후확률이라고 주장하지 않으며, 모든 수식은 코드(`nearzero/risk/*`)와 일치합니다.

## 1) 쌍 위험 구성 요소 (모두 [0,1] 단조 함수로 정규화)

### 충돌확률 `P_c`
앙상블 샘플 집합 `{a_i}, {b_i}` (i=1..N 개별 궤적)에서의 최소거리 관점:

```
d_i(t) = || a_i(t) - b_i(t) ||          (t = 0..4s, 0.1s 간격)
min_d_i = min_t d_i(t)
P_c = mean_i( min_d_i < r_sum )           (r_sum = 물리반경 합: 사람 0.35 + 차량 0.60…)
```

- `component_collision(P_c) = clip(P_c, 0, 1)`
- `samples=None`(불확실성 제거)이면 기대 궤적 거리로 0 또는 1.

### TTC (Time To Collision)
확정 궤적의 거리 프로파일 `d(t)`가 반경 합 `r_sum` 이하로 들어가는 첫 시점
(`risk/collision.py: ttc_from_profiles`). 카운터팩추얼 하에서도 동일 함수 사용 →
"회피 후에도 낡은 TTC가 남는" 문제를 원천 차단.

```
component_ttc(ttc) = 1 - sigmoid(ttc; x0=2.5, k=1.4)
```

### PMD (Predicted Minimum Distance)
기대 궤적 간 최소거리.

```
component_pmd(pmd) = 1 - sigmoid(pmd; x0=1.5, k=3.0)
```

### 구역 침입 / 규칙 위반
- `zone_intrusion_probability`: 샘플이 제한구역 폴리곤에 들어가는 비율.
- 규칙 위반: `severity_value(0.35/0.65/1.0) × violation_intensity(거리 초과 비율)`.

## 2) 결합 (sigmoid 파라미터=엄격도)

```
score = 100 × Σ_k w_k·component_k / Σ_k w_k        (0..100)
        w: collision 1.0, ttc 0.9, pmd 0.7, zone 0.6, rule 0.5 (configs)
```

## 3) 상태 분류 임계 (configs: RiskConfig)

| 상태 | 기준 |
|---|---|
| SAFE | score < 25 |
| CAUTION | < 50 |
| WARNING | < 75 |
| CRITICAL | ≥ 75 |

`warn_risk=45.0`에서 사전 경보(PREDICTED 이벤트), `intervention_risk=48.0`에서 counterfactual 추천.

## 4) 신뢰도 (confidence)
트랙 히트 수와 충돌확률 크기에 의한 경험적 신뢰:

```
confidence = clip( 0.35 + 0.03×hits + 0.25×P_c , 0, 1 )
```

- LOW  < 0.35 · MEDIUM < 0.7 · HIGH 이상
- 추천 발생 요건: `score ≥ intervention_risk`, `pair.score > warn_risk`,
  `confidence ≥ verification.required_confidence` (configs 기본 0.45).

## 5) Counterfactual 개입 물리 (정직한 모델)
`planner/interventions.py`

| 개입 | 적용 |
|---|---|
| HUMAN_STOP | 사람이 반응 0.7s 후 1.5 ㎧ 감속으로 감속 (평균속서계수 적분) |
| HUMAN_LEFT / RIGHT | 사람 횡이동 속도 = `lateral_shift(1.2 m) / dodge_time(0.6 s)` = 2 ㎧ |
| STOP_VEHICLE | 차량이 반응 0.7s 후 `stop_decel(2.0 ㎧²)` 로 정지 (래프 적분) |
| SLOW_VEHICLE_50/30 | 속도계수 0.5/0.3 |
| VEHICLE_LEFT / RIGHT | 차량 횡이동 2 ㎧ ... |
| NO_ACTION | 무변화 (현상 유지 = 비교 기준) |

**효용 선택:**

```
reduction_norm = (현재점수 - resulting_risk) / 100
utility = reduction_norm × (1.5 if achieved else 1.0) - cost_weight(0.35) × cost
```
- `achieved` = 개입이 P_c 를 실제로 낮췄는지 (resulting risk 기준)
- 사소한 HUMAN_LEFT tie-break(+0.005). 상대 비교로만 쓰이는 목적함수.

**물리적 정직성 예시** (`demo_avoided`, t≈1.9s):
보행자가 0.9 ㎧로 걸어오는데 차량이 정지하면, 보행자는 계속 걸어와 **정지한 카트에 충돌** →
STOP_VEHICLE 은 여전히 high risk(43). 반면 VEHICLE_RIGHT(차선 우회)는 충돌 회피 —
resulting risk = 1. 즉 시스템은 "정지는 안전"이라는 일반론이 아니라 **이 기하에서 정말
효과적인 행동**을 골라냅니다.

## 6) 검증(VERIFY) 조건
- 추천 후 위험 ≤ `resolve_drop(0.5) × initial_risk` 인 프레임 5회 연속 → **PREVENTED**
- 4프레임 이상 객체 소실 → FALSE_ALARM / TTC < 0 → UNRESOLVED (구현: ttc≤0 분기)

## 7) 공개된 한계
- 시나리오는 2D 지면 가정; 높이/각도 정보 없음.
- 확률은 정성적 보정 추정 — 캘리브레이션 데이터로 더 엄밀하게 만들 수 있는 여지 명시.
- 자율 급제동, 로컬라이제이션 오차, 통신 지연 등은 시뮬레이션 범위 밖.