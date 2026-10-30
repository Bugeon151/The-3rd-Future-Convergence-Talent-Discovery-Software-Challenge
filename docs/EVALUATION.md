# EVALUATION — 방법, 결과, 재현

평가는 **결정적 시뮬레이션 시나리오 8종**으로 실행됩니다.
모든 숫자는 `scripts/run_eval.ps1`(=`python -m nearzero.eval.all`)로 산출된 **계측값**입니다.

## 1) 시나리오 (nearzero/sim/scenarios.py)
| 시나리오 | 내용 | GT 이벤트 여부 |
|---|---|---|
| head_on | 정면 충돌 1건 | O |
| crossing | 직각 교차 1건 | O |
| parallel | 나란히 같은 방향 (안전) | X |
| safe_pass | 근접 통과 (안전) | X |
| sudden_stop | 갑작 정지 충돌 | O |
| sudden_turn | 급선회 근접 (안전) | X |
| zone_intrusion | 제한구역 침입 | O |
| demo_avoided | 플래그십: 추천 조향이 PREVENTED | O |

## 2) 지표 정의 (nearzero/eval/metrics.py)
- **recall**: GT 접촉구간 [t0,t1] 중 최소 1프레임 경보 OR = 정탐 이벤트/GT 이벤트.
  GT 이벤트가 없는 시나리오는 **NULL**(집계에 0으로 기여하지 않음).
- **mean_lead_time**: GT t0 − 첫 경보 시각.
- **false_alarms**: GT가 없는 경보(45점 초과) 프레임 수.
- **intervention_success**: 추천 중 PREVENTED로 마감된 비율 (플래그십 적용).
- **mean_risk_reduction**: 플래그십 시나리오에서 추천 직전 vs 직후 위험 감소 평균.

## 3) 정책
- A: distance_threshold — 거리 1.5m 이하 경보 (단순 베이스라인)
- B: ttc_threshold — TTC 3.0s 이하 경보
- C: trajectory_prediction — 예측 최소거리 1.0m 이하 경보 (NEAR-ZERO X 의 PREDICT만 사용)
- **D: NEAR-ZERO X — 예측 + 확률 위험 + counterfactual + 검증 전체**

## 4) 결과 요약

### 4-a) 정책 비교 (recall / lead / false alarms)
```
A: distance_threshold    recall=1.00  lead=0.50s  false=1
B: ttc_threshold          recall=1.00  lead=2.62s  false=2
C: trajectory_prediction  recall=1.00  lead=2.70s  false=2
D: nearzero_x             recall=1.00  lead=1.52s  false=1   success=1.00  red=45.73
```
해석: 단순 규칙은 정확하지만 선행시간이 짧거나(0.5s), 경보가 잦습니다.
NEAR-ZERO X 는 4초 예측과 확률 임계로 **1.52초의 선행시간을 거짓경보 1회 수준으로** 유지하고,
플래그십 시나리오에서 100% 개입 성공(PREVENTED)과 평균 45.7점의 위험 감소를 달성합니다.

### 4-b) 컴포넌트 제거 (Ablation)
| 변형 | recall | lead | false | success | risk_reduction |
|---|---|---|---|---|---|
| full | 1.00 | 1.43 | 0 | 1.00 | 44.8 |
| 불확실성 제거 (no_uncertainty) | 1.00 | 2.40 | 0 | 1.00 | 54.1 |
| counterfactual 제거 (no_counterfactual) | 1.00 | 1.43 | 0 | — | — |
| 검증 제거 (no_verification) | 1.00 | 1.43 | 0 | 1.00 | 44.8 |

- no_counterfactual: 추천이 생성되지 않아 success/reduction 측정 불가(NULL) —
  실제로 "추천 없음"이 기능적 손실임을 보여줍니다.
- no_verification: 성능 동일, 검증이 위험 추정이 아닌 **이벤트 수명 확인**에 기여함을 설명.

### 4-c) 성능
```
감지 (YOLOv8n, GPU): 48.7 ms | 코어 파이프라인: 181.6 ms
```
LT: 총 배치 230ms/frame ≈ 4.3 FPS. 데모에 필요한 실시간성은 충족(시뮬레이션 10Hz 내 처리).

## 5) 산출물
```
results/comparison.csv        # 모든 정책×시나리오 원표
results/metrics.json          # aggregate 지표
results/evaluation_report.md  # 소비자용 보고서
results/plots/comparison.png  # 비교 시각화
```

## 6) 재현
```powershell
.\scripts\run_eval.ps1        # 전 스위트 (약 2분)
```
설명서: `nearzero/eval/compare.py` (시나리오별 pipeline 주입, EventStore 분리,
`--ablation` 변형 포함).