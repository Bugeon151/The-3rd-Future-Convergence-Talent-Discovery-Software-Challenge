# ROADMAP — 향후 계획

## 현재 (v1, 완료)
- [x] PERCEIVE: YOLOv8n + Kalman 지면 추적
- [x] PREDICT: 상호작용 인지 앙상블 4초 예측 (불확실성 샘플)
- [x] PREVENT: 확률 위험 + counterfactual 개입 탐색
- [x] VERIFY: PREVENTED 이벤트 검증 + SQLite 저장
- [x] 평가: 정책 비교 8시나리오, 성능 벤치, ablation
- [x] 대시보드 + 헤드리스 데모 + 테스트 38개

## 단계 1 — 실제 하드웨어 (우선순위 최고)
- 실제 이동로봇(카트)에 탑재: `SimRunner` 대신 ROS/직렬 모터 브리지 `Executor`
  - `execute_vehicle_dodge(...)` 신호 → 모터/GPIO/ROS topic
- 카메라 보정: 정면 1대 + 캘리브레이션 툴 자동화 (chessboard)
- 필드 실증: 5장면 × 2분 촬영으로 error case 수집

## 단계 2 — 모델 개선
- 예측 백본 교체: `EnsembleForecaster` → Transformer 스냅샷(경량), 동일 인터페이스
- 불확실성 보정: 정규화 흐름(Normalizing Flow) 또는 MC-Dropout, 달력 기반 교정 검증
- 다중 센서 융합: LiDAR/IMU 추가, 상태(위치·방향·속도) 필터 확장

## 단계 3 — 배포 규모
- 멀티 카메라 병합(Homography 파이프라인 2대), 구역 관리 UI
- 위험 예측 오류(≈ 45점 근처) 시 관리자 채널; 사중 안전 캐스케이드 경고 (사전-중-사후)
- 실시간 통신 프로토콜(MQTT/WebRTC), 10Hz→30Hz 최적화 (양자화/ONNX)

## 단계 4 — 공개·인증
- 데이터셋 공개 (합성 GT 포함), 성능 리포지트 생성기 자동화
- ISO 61508 / ISO 12100 (기계 안전) 준수 검토, 외부 감사용 결정적 로그 포맷

## 비용 예상 (경진대회 수준으로는 초기 단계)
- 1단계: 로봇 1기 + 카메라 + 마이크로 컨트롤러 ≈ 소액
- 2단계: GPU 1기(keras) 추가 학습 ≈ 시간 자원 중심