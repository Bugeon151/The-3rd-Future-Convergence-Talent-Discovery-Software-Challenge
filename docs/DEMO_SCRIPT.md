# DEMO SCRIPT — 플래그십 시연 `demo_avoided`

## 준비
```powershell
.\scripts\setup.ps1       # 최초 1회
.\scripts\run.ps1         # 대시보드 + 시뮬레이션 (http://localhost:5000)
```
헤드리스 검증(발표 전 1번 실행해 트랜스크립트 저장):
```powershell
.\scripts\run_demo.ps1 > demo_transcript.txt
```

## 시나리오 (사람 + 카트)
- 보행자 `P0`: (1.0, 5.0)에서 +x 방향으로 0.9 ㎧ 보행
- 카트 `V0`: (9.5, 5.0)에서 −x 방향으로 1.9 ㎧ 주행
- 두 객체가 같은 복도(y=5.0)에서 **정면 회피 불가** 상황을 만들기 위함

## 예상 흐름 (계측된 실제 수치)
| t (s) | 상황 | 위험 | 비고 |
|---|---|---|---|
| 0~1.5 | 멀리 접근 | < 10 | SAFE/CAUTION |
| ~1.9 | 근접 | **49** (충돌확률 72%, TTC 2.64s) | CAUTION(≥50=WARNING), 추천 트리거(≥48) |
| 1.9 | 추천 | `VEHICLE_RIGHT` resulting risk = 1 | >> STOP_VEHICLE(43), NO_ACTION(49) |
| 1.9+ | **세계에서 조향 실행** | 6 → 3 → 0 | committed lane shift |
| 이후 | 검증 | `PREVENTED` 1건 | 이벤트 저장 |

헤드리스 출력 예시 (물론 실행 시 실측치가 결정적으로 나옴):
```
peak risk snapshot (t=1.9)
  state=CAUTION  risk=49/100
  collision_probability=72%
  tracked evictions: [t over]  events={'PREVENTED': 1}
  counterfactual table:
   VEHICLE_RIGHT   risk=  1   reduction= 48 <-- BEST
   VEHICLE_LEFT    risk=  1   reduction= 48
   HUMAN_LEFT      risk=  8   reduction= 41
   ...
   NO_ACTION       risk= 49   reduction=  0
  executed maneuver: VEHICLE_RIGHT at t=1.9s
  verification: PREVENTED on event #1 (recommended=VEHICLE_RIGHT, peak_risk=49)
```

## 리허설 체크리스트
1. `run.ps1` 실행 → 브라우저 자동 연결? (Flask가 콘솔에 URL 출력)
2. 좌측 "시나리오" 드롭다운에서 demo_avoided 선택 → 0.1초 간격 재생
3. 위험 게이지 0→49 상승, BEST 카드에 VEHICLE_RIGHT
4. **조향 실행 애니메이션** (카트가 오른쪽 1.2m 이동) → 위험 6→0
5. 이벤트 테이블에 `PREVENTED` 1건 확인
6. 규칙 예시 추가(Ko/En) → 위반 배지 변화 확인
7. 수동 개입 버튼(HUMAN_LEFT 등) → 즉시 위험 게이지 반응

## 트러블슈팅
- GPU 없음: `setup.ps1`에서 torch CPU 설치 경로로 대체, 감지 속도 저하만 있음.
- 5000 포트 사용 중: `python -m nearzero.demo --port 5001`.
- 웹캠 데모: `python -m nearzero.demo --mode webcam` (사람/차량이 카메라에 보여야 함).