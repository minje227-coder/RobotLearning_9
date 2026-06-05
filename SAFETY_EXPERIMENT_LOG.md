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
- 실행 env(`lerobot`)·GPU·robot geom 네이밍 확인 작업 진행.
- `hannuri` 브랜치에 `main` 병합 완료 후 푸시. 본 로그 파일 생성.
- (다음) Phase 1 하네스(`Test/run_safety_eval.py`) 작성 → 베이스라인 측정.
