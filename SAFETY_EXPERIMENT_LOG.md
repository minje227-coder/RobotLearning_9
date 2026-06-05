# VLSA-AEGIS 듀얼암 안전성 실험 — 진행 로그

> 이 문서는 듀얼암 VLSA-AEGIS 안전성 실험의 작업 history를 기록한다.
> 브랜치: `hannuri` (origin: `minje227-coder/RobotLearning_9`). 작업 서버: `kanu` (`~/workspace/RobotLearning_9`).

---

## 0. 배경 / 프로젝트 정체

- **원본 연구**: VLSA (Vision-Language-Safe Action) / **AEGIS** — Tsinghua, arXiv 2512.11891.
  사전학습 VLA 정책 위에 **재학습 없이 끼우는 CBF 기반 안전 레이어(SC layer)**. 정책 액션을 CBF-QP로 최소 보정해 충돌 방지. (단일 Franka, 정적 장애물, π0.5, SafeLIBERO 벤치)
  - 원본 코드: `THU-RCSCT/vlsa-aegis` (이 repo의 `vlsa-aegis/` 하위와 동일).
- **이 프로젝트(RobotLearning_9)의 확장**: 원본을 **듀얼암**으로 확장.
  - 두 Franka Panda가 테이블 양 끝에서 마주봄 (`LIBERO_Dual_Tabletop_Manipulation`).
  - 상대팔 EE를 **동적 타원체 장애물**로 모델링 (+ 정적 장애물).
  - 정책: π0.5 대신 **자체 학습한 SmolVLA** (robot0/robot1 각각).
  - 태스크: milk → trash can 안 target.

## 1. 현재까지 완료된 것

- **데이터 생성**: `Data generation/` — IK 스크립트 데모로 성공 에피소드 수집 → LeRobot v3 변환 → HF `Whalswp/vlsa_robotX_vY` 업로드. 관측: 3카메라(side_left/wrist/side_right, 256²) + state 8D(관절7+그리퍼), 액션 8D(관절델타7+그리퍼), 10Hz.
- **정책 학습**: SmolVLA 6개 런 (`lerobot/smolvla_base` 파인튜닝, bs32, lr1e-4, chunk50). 산출물은 `~/workspace/vlsa_smolvla_robot{0,1}_v{2,3,4}/` (git 밖).

| 런 | dataset | steps | 최종 train/loss |
|---|---|---|---|
| robot0_v2 | vlsa_robot0_v2 | 100k | 0.0046 |
| robot0_v3 | vlsa_robot0_v3 | 75k | 0.0062 |
| robot0_v4 | vlsa_robot0_v4 | 30k | 0.0069 |
| robot1_v2 | vlsa_robot1_v2 | 100k | 0.0088 |
| robot1_v3 | vlsa_robot1_v3 | 75k | 0.0033 |
| robot1_v4 | vlsa_robot1_v4 | 30k | 0.0053 |

- **정성 검증**: `Test/test_model_grasp.py` 로 단일 에피소드 롤아웃 영상(`Test/grasp_test_v4_*.mp4`).

## 2. 아직 안 된 것 (= 이번 작업 목표)

논문 Table 1에 대응하는 **듀얼암 정량 실험**이 없음. TODO.md의 매트릭스 미수행:

- VLSA 0개 (양팔 충돌, 실패 예상)
- VLSA 1개 (한 팔만 안전)
- VLSA 2개 (양팔 안전)
- × {bimanual, bimanual+장애물}

→ 성공률(TSR)·충돌회피율(CAR)을 집계하는 **측정 하네스 + 듀얼암 CBF 안전 레이어**를 만들어야 함.

## 3. 인프라 현황 (실측)

- **실행 env**: conda `lerobot` (py3.12) — robosuite·lerobot·imageio·cv2·torch·pandas 모두 있음. (`libero` env는 비어있음)
- **GPU**: kanu A4000 × 8, 전부 idle.
- **기존 하네스 한계**: `test_model_grasp.py`는 단일 seed → MP4 1개 + `final_success` 출력만. 다중 에피소드·충돌지표·집계 없음.
- **CBF 미연동**: `compute_h_coeffs`/타원체/QP는 원본 `vlsa-aegis/main/main_aegis.py`(단일팔, EE속도/OSC, π0.5)에만 존재. 듀얼 env엔 없음.
- **핵심 미스매치**: SmolVLA는 **관절공간** 출력인데 논문 CBF는 **Cartesian EE 속도** 보정 → Jacobian 역매핑 필요(env에 `get_site_jacp` 있음).

## 4. 결정 사항

- **진행 범위**: Phase 1 먼저 (측정 하네스 + 베이스라인 VLSA=0).
- **체크포인트**: v4 (robot0_v4 / robot1_v4, 30k).
- **충돌 판정**: mujoco 접촉 기반 (robot0 geom ↔ robot1 geom 접촉).
- **단계 계획**:
  - Phase 1 — 다중 seed 롤아웃 + 팔-팔 충돌검출 + CAR/TSR CSV (베이스라인).
  - Phase 2 — 듀얼암 CBF(상대팔=동적 타원체) + Jacobian 관절 역매핑 (VLSA 1/2개).
  - Phase 3 — 정적 장애물 변형 (GroundingDINO/타원체), 매트릭스 완성.

## 5. Git / 브랜치 셋업

- kanu `~/workspace/RobotLearning_9` ↔ `minje227-coder/RobotLearning_9` 연동(SSH origin).
- 작업 브랜치 = **`hannuri`**. `main`(21커밋 앞섬)을 hannuri로 병합(merge commit `910f44f`)해 최신 코드 확보. 충돌 3건(`auto_scene.py`, `AUTO_SCENE.mp4`, `trash_can.xml`)은 main 버전 채택.
- 이 로그(`SAFETY_EXPERIMENT_LOG.md`)는 hannuri 브랜치에 커밋·푸시하여 동기화.

---

## 진행 로그 (timeline)

### 2026-06-05
- 프로젝트/논문/원본 repo 숙지 완료. 6개 SmolVLA 런 + 데이터 파이프라인 + AEGIS CBF 구조 파악.
- 실험 인프라 현황 점검: 측정 하네스·듀얼 CBF 부재 확인. 관절공간 vs Cartesian CBF 미스매치 식별.
- 실행 env(`lerobot`)·GPU·robot geom 네이밍 확인. geom: `robot0_*`/`robot1_*`/`gripper{0,1}_*`, 각 66개.
- `hannuri` 브랜치에 `main` 병합(`910f44f`) 후 푸시. 본 로그 파일 생성.
- **Phase 1 하네스 작성**: `Test/run_safety_eval.py` 신규.
  - `test_model_grasp.py`의 안정 헬퍼 재사용 + 에피소드 드라이버(스크립트 grasp→정책 핸드오프) 재구현.
  - 다중 seed 루프, **팔-팔 충돌검출**(`arm_arm_contact`: contact의 두 geom이 robot0↔robot1이면 충돌), TSR/CAR/safe-success 집계 → CSV+JSON.
  - 실행: `lerobot` env + `MUJOCO_GL=egl`, 정책 1회 로드 후 에피소드마다 reset.
- **충돌검출 검증**: 스모크 3 에피소드 모두 충돌. 접촉 geom이 `gripper0_hand_collision|gripper1_hand_collision`, `gripper0_hand_collision|robot1_link7_collision` 등 실제 팔-팔 접촉으로 확인(오탐 아님). 충돌 step seed별 상이(30/36/93).
- **베이스라인(VLSA 0) 완료**: robot0_v4/robot1_v4, seeds 1~50, max-steps 400 → `Test/results/baseline_v4.{csv,json}`.

#### 결과 — VLSA 0개 (안전장치 없음), N=50

| 지표 | 값 |
|---|---|
| **TSR** (task success rate) | **4%** (2/50) |
| **CAR** (collision avoidance rate) | **4%** (2/50 무충돌, 즉 48/50 충돌) |
| **safe-success** (성공 & 무충돌) | **0%** |

- 충돌 step: min 28 / median 91 / max 177 (스크립트 grasp 이후 정책 구간에서 주로 발생).
- 성공 seed: 8, 26 (정책 자체는 성공 가능 → 성공 지표 유효). 단 둘 다 충돌 동반.
- 충돌 부위 분포(48건 중): 그리퍼-그리퍼 34, 나머지는 link5/6/7 등 팔 상부 클래시. → **두 팔이 중앙 trash can으로 동시에 접근하며 손끝이 부딪히는 게 지배적 실패 모드.**
- **해석**: 안전 레이어 없이 두 정책을 동시에 돌리면 96%에서 팔-팔 충돌, safe-success 0%. Phase 2(CBF)의 대조군으로 충분히 강한 베이스라인.

- Phase 2 — 듀얼암 CBF(상대팔=동적 타원체) + Jacobian 관절 역매핑 구현·측정 (아래).

### 2026-06-05 (Phase 2)
- **CBF 구현** `Test/cbf_safety.py`: 상대팔 EE를 타원체(semi-axes 0.06/0.12/0.11 + margin)로 보고, 중심연결선 방향 반경으로 `h = dist − r_self − r_other − margin`. 이산 CBF half-space projection으로 EE 변위 보정 → **damped Jacobian 역매핑**으로 관절 보정(정책 의도 보존, 위반 시에만 개입). QP 솔버 불필요(단일 선형제약 closed-form).
- **action_scale 보정**: JointPositionController가 1 control step에 목표 delta의 일부만 실현 → 정상상태 측정으로 `action_scale=0.166` 확정(J@dq → 실제 EE 변위). 방향 cos≈1.
- **하네스 통합**: `run_safety_eval.py --safety-arms {none|0|1|0 1}` (VLSA 0/1/2), 개입수·min_h 기록, 영상 옵션.
- **충돌검출 영상**: VLSA-0 대표 seed(1,3 충돌 / 8,26 성공) → `Test/results/videos/VLSA0_v4_seed*.mp4`.

#### 결과 — VLSA 매트릭스 (robot0_v4/robot1_v4, seeds 1~50, max-steps 400)

| 조건 | safety arms | TSR | CAR(전체) | CAR(정책구간) | 평균 개입수 | grasp단계 충돌 |
|---|---|---|---|---|---|---|
| **VLSA-0** (없음) | – | 4% | 4% | 16% | 0 | 6 |
| **VLSA-1** (arm0) | [0] | 0% | 42% | 54% | 28.9 | 6 |
| **VLSA-2** (양팔) | [0,1] | 0% | 64% | 76% | 50.1 | 6 |

- **핵심**: CBF가 충돌을 단조적으로 크게 줄임. CAR 4%→42%→64% (VLSA-2 = 16× baseline). 정책구간 한정 CAR은 16%→54%→76%로 CBF 효과가 더 뚜렷. → 논문 핵심 가설("VLSA가 충돌 회피율을 높인다")의 듀얼암 버전 **검증 성공**.
- **VLSA-1의 한계**: 한쪽만 회피하면 상대가 밀고 들어와 절반가량만 회피(비대칭, 4-seed 검증서 arm0 0% vs arm1 50%). 머리 맞댄 head-on에선 양팔 모두 안전(VLSA-2)이 필요.
- **grasp단계 충돌 6건은 세 조건 공통**: 스크립트 grasp 구간(정책·CBF 적용 전) 충돌이라 구조적으로 CBF가 못 막음 → CAR 상한을 제약. (정책구간 CAR이 더 공정한 CBF 지표)
- **TSR 트레이드오프 (4%→0%)**: 양팔이 **같은 중앙 trash can**에 넣어야 해서, 강한 상호회피가 중앙 근접을 막아 성공률이 떨어짐. 논문 5.3의 "safety-induced distribution shift"가 공유목표 기하로 증폭된 사례. baseline TSR도 4%로 이미 매우 낮음(동시 비협조 정책의 태스크 난이도).
- **CBF 파라미터**: margin 0.04, alpha 0.5, scale 0.166, damping 0.05.

#### 한계 / 다음 후보
- TSR 회복: 시작시점 stagger(이미 `_delay_ver` 실험 흔적), 또는 trash can 분리/순차 배치, margin·alpha 튜닝으로 안전↔성공 균형점 탐색.
- grasp단계도 CBF로 보호(현재 정책구간만).
- Phase 3: 정적 장애물(GroundingDINO/타원체) 추가한 매트릭스.
- 산출물: `Test/results/{baseline_v4,vlsa1_v4,vlsa2_v4}.{csv,json}`, `cbf_safety.py`, `run_safety_eval.py`.
