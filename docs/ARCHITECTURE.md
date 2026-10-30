# ARCHITECTURE

NEAR-ZERO X 는 4단계 폐쇄 루프(closed loop) 구조입니다.
한 프레임의 라이프사이클: `SafetyPipeline.step()` 하나의 호출 안에서 진행됩니다.

```
                   ┌────────────────────────────────────────────┐
   동영상/카메라     │   nearzero/app/pipeline.py (SafetyPipeline) │
  ──────────▶        │                                            │
                   │  [1] PERCEIVE                               │
                   │   - 객체 감지 (YOLOv8n / 시뮬레이션)          │
                   │   - 지면(Homography) 투영, 픽셀→월드 좌표      │
                   │   - Kalman 필터 추적 (ID 유지, 속도 추정)     │
                   │                                            │
                   │  [2] PREDICT                                │
                   │   - 상호작용 인지 앙상블 궤적 예측 (4s / 0.1s) │
                   │   - 불확실성 공분산 → 샘플 집합 (N=60)        │
                   │                                            │
                   │  [3] PREVENT                                │
                   │   - 쌍(pair) 위험 점수 (충돌확률·TTC·PMD·구역  │
                   │     ·규칙) → 0..100                         │
                   │   - counterfactual: 개입별 "만약 ...라면       │
                   │     resulting risk" 시뮬레이션 → 최적 추천     │
                   │   - 규칙 위반 감지 (결정적 DSL)               │
                   │                                            │
                   │  [4] VERIFY                                 │
                   │   - PREDICTED→RECOMMENDED→PREVENTED         │
                   │     (낮은 위험 유지·객체 소실·TTC 임계로 종결)  │
                   └──────────────┬─────────────────────────────┘
                                  │ Snapshot (JSON-safe)
                                  ▼
              Flask API (/api/state, /video_feed, /api/events …)
                                  │
                                  ▼
                     웹 대시보드 (frontend/static)
```

## 모듈별 상세

### PERCEIVE — `vision/`, `core/`
- 감지: Ultralytics YOLOv8n (`vision/detector.py`). pure-PyTorch NMS fallback 내장
  (torchvision::nms 가 없는 환경에서도 동작).
- 지면 투영: `Homography.to_world(px, py)` — 캘리브레이션 점 4개로부터 계산.
- 추적: `core/tracks.py` TrackManager — IoU+거리 결합 매칭, 미검출 프레임은 Kalman 예측으로 보간
  (최대 25프레임), `min_hits=3` 이후 성숙 트랙으로 승격.
- 프라이버시: `vision/privacy.py` — 검출 얼굴 영역 즉시 모자이크 처리.

### PREDICT — `prediction/forecast.py`
- `EnsembleForecaster.forecast(pos, vel, cov, t)` → `Trajectory(positions, samples)`.
- `n_samples=0` (불확실성 제거) 시 확정 궤적(`samples=None`)만 반환 — `.std()` guard와 함께
  nondeterministic 경고 제거.

### PREVENT — `risk/`, `planner/`
- `RiskEngine.evaluate(contexts, zones, violations)` → ScenarioRisk(쌍 위험 목록).
- `planner/counterfactual.py` — 7+1개 개입(`NO_ACTION, HUMAN_STOP, HUMAN_LEFT/RIGHT,
  STOP_VEHICLE, SLOW_VEHICLE_50/30, VEHICLE_LEFT/RIGHT`)을 궤적에 반영해 재추론.
  효용 = `(위험감소/100) × (달성 시 1.5) − 비용가중치×비용` + 미세bias.
- 개입 물리(`planner/interventions.py`): 감속은 반응시간+래프으로 적분, 횡이동은
  `lateral_shift/dodge_time` 속도로 적용. 정직한 물리 — 정지가 불가능한 기하에서는
  정지 개입이 낮은 점수를 얻도록 설계됨.

### VERIFY — `verify/monitor.py`
- 이벤트 수명: `PREDICTED → WARNING → RECOMMENDED → PREVENTED`(또는 `FALSE_ALARM`).
- 종결 조건: 추천 후 위험 `resolve_drop×initial` 이하로 떨어져 5프레임 유지 → **PREVENTED**;
  객체 소실 4프레임 → FALSE_ALARM; TTC ≤ 0 → UNRESOLVED.
- 모든 전이를 `nearzero/events/store.py`(SQLite, thread-safe lock)에 기록.

### 규칙 — `rules/`
- 자연어 문장(한국어/영어) → 결정적 패턴 파서(`rules/natural.py`) → 스키마 검증(`rules/schema.py`)
  → JSON DSL(`configs/rules.json`) → `rules/engine.py`가 거리 조건으로 위반 강도(intensity) 계산.
- 안전 판단 경로에 LLM이 개입하지 않음.

### 러너/서버 — `server/`
- `SimRunner`: 사전 정의 시나리오를 시간순 재생. 위험 ≥ `intervention_risk`이고 최적 개입이
  `VEHICLE_*`이면 **세계의 시간축에서 그 조향을 실제로 실행**(committed lane shift) →
  프레임들이 있는 그 개입효과를 측정 → 검증 이벤트가 PREVENTED로 마감. (데모 핵심 폐쇄 루프)
- `CameraRunner`: 웹캠(`--mode webcam`) 또는 동영상(`--mode video`) 실시간 처리.
- Flask API는 JSON 상태 스냅샷 + MJPEG 스트림을 제공.

## 데이터 흐름 (프레임 단위)

```
Frame/Objects ─▶ TrackManager.step ─▶ [Track + Kalman cov]
                    │
                    ├─▶ RuleInput ─▶ rules.evaluate_rules ─▶ violations
                    │
                    └─▶ EnsembleForecaster ─▶ TrackContext.traj
                                                   │
                    RiskEngine.evaluate ──────────▶ pair risks + scenario risk
                                                   │
                    (점수 ≥ intervention_risk) ─▶ CounterfactualPlanner.plan
                                                   │
                    VerificationMonitor.update(risk, rec, best, alive)
                                                   │
                    Snapshot ─▶ snapshot_to_state ─▶ GET /api/state
```

## 확장 지점
- 실제 로봇 적용: `Runner`만 교체하고 `execute_vehicle_dodge` 신호를 모터 CAN/ROS 토픽으로 연결.
- 멀티 카메라: Homography 캘리브레이션을 카메라별 저장 후 병합.
- 고급 예측: `EnsembleForecaster`를 Transformer/확산 모델로 교체 가능 (인터페이스 동일).