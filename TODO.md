# Project TODO

> 구현 메모:
> - 학습: 로봇 1개 (`preview_dual.py 1`) → single policy 학습
> - 추론/벤치: 로봇 2개 (`preview_dual.py 2`) → 각자 단일 policy 실행, 한쪽에만 VLSA
> - 환경: `LIBERO_Dual_Tabletop_Manipulation` (1~2 로봇 모두 지원)

---

## Data Generation

- [ ] 로봇팔 환경, 객체 세팅
  > **담당 파일:**
  > - `Data generation/preview_dual.py` — 환경 렌더링·MP4 저장 스크립트 (CLI 인수로 로봇 수/객체 위치 조정)
  > - `vlsa-aegis/safelibero/.../problems/dual_tabletop_manipulation.py` — `LIBERO_Dual_Tabletop_Manipulation` 클래스 (로봇 2개 배치, side table 추가)
  > - `Data generation/dual_scene.bddl` — 기본 씬 정의 (실행 시 동적으로 덮어씀)

- [ ] 시나리오마다 객체 랜덤 재배치
  > **담당 파일:**
  > - `Data generation/preview_dual.py` — `--basket-x/y`, `--milk-x/y`, `--milk-ref` 인수로 범위 지정 후 랜덤 샘플링
  > - `dual_tabletop_manipulation.py` — `SIDE_DEPTH`, `SIDE_WIDTH`, `SIDE_Y_OFFSET`, `SIDE_X_OFFSET` 모듈 변수로 side table 위치 조정

- [ ] 객체위치 받아와서 Data 수집하는 script 작성 *(LeRobot V3.0 format 추천)*
  > **담당 파일:** *(미구현 — 신규 작성 필요)*
  > - `Data generation/preview_dual.py`에서 env obs를 받아 LeRobot 포맷으로 저장하는 로직 추가
  - [ ] Joint Space

---

## Train

- [ ] SmolVLA (LeRobot)
  > **담당 파일:**
  > - `vlsa-aegis/scripts/train.py` — JAX 기반 학습 엔트리포인트 (pi0/pi05)
  > - `vlsa-aegis/scripts/train_pytorch.py` — PyTorch DDP 학습 엔트리포인트 (단일/멀티 GPU)
  > - `vlsa-aegis/scripts/compute_norm_stats.py` — 학습 전 데이터 정규화 통계 계산
  > - *(SmolVLA용 config/adapter는 미구현 — 별도 추가 필요)*

---

## Bench 환경 구성

- [ ] 기존 환경에 로봇팔 2개
  > **담당 파일:**
  > - `Data generation/preview_dual.py` — `python preview_dual.py 2` 로 듀얼 모드 실행
  > - `dual_tabletop_manipulation.py` — `_load_model`에서 robot1을 반대편(x = +0.46, rot=π)에 배치

  - [ ] 각자 단일 policy, 1개 로봇에만 VLSA 적용
    > **담당 파일:**
    > - `vlsa-aegis/scripts/serve_policy.py` — 학습된 체크포인트에서 WebSocket 정책 서버 실행 (포트별로 2개 띄울 것)
    > - `vlsa-aegis/main/main_aegis.py` — VLSA 포함 추론 루프 (한 로봇에 VLSA 적용)
    > - `vlsa-aegis/main/utils.py` — CBF·Ellipsoid·장애물 감지 유틸 함수 모음

    - [ ] VLSA Input (2개)
      - [ ] 반대편 로봇팔 EE의 Ellipsoid [Dynamic] (TF 변환) → 실시간
        > **담당 파일:**
        > - `vlsa-aegis/main/utils.py` — `fit_ellipse`, `get_point_cloud`, `filtering_points`, `compute_h_coeffs_3d`
        > - `vlsa-aegis/main/main_aegis.py` — 실시간 EE pose 읽어 Ellipsoid 생성 및 CBF 적용 루프
      - [ ] 장애물 [Static] (Grounding DINO)
        > **담당 파일:**
        > - `vlsa-aegis/main/utils.py` — `obstacle_detection` (Grounding DINO 호출)
        > - `vlsa-aegis/main/main_aegis.py` — `OBSTACLE_POS`, `OBSTACLE_RADIUS` 상수 + CBF 제약 계산

    - [ ] 시나리오마다 객체 재배치 + 장애물 재배치
      > **담당 파일:** *(미구현 — 신규 작성 필요)*
      > - `Data generation/preview_dual.py` 또는 새 bench 스크립트에서 에피소드마다 재배치 로직 추가

---

## Experiments (VLSA 점차 늘리기: 0 → 1 → 2)

### VLSA 아예 없는 버전
> **실행 방법:** `serve_policy.py` 2개 + VLSA 없이 단순 policy rollout 스크립트
- [ ] bimanual test *(실패 예상)*
- [ ] bimanual + 장애물 test *(실패 예상)*

### 로봇팔 1개만 VLSA
> **실행 방법:** `vlsa-aegis/main/main_aegis.py` (한 로봇에만 VLSA 입력)
- [ ] bimanual test *(성공)*
- [ ] bimanual + 장애물 test *(실패 예상)*

### 로봇팔 2개 다 VLSA
> **실행 방법:** `vlsa-aegis/main/main_aegis.py` (양쪽 로봇 모두 VLSA 입력)
- [ ] bimanual test *(성공)*
- [ ] bimanual + 장애물 test *(성공)*
