# Project TODO

## Data Generation
- [ ] 로봇팔 환경, 객체 세팅
- [ ] 시나리오마다 객체 랜덤 재배치
- [ ] 객체위치 받아와서 Data 수집하는 script 작성 *(LeRobot V3.0 format 추천)* 
  - [ ] Joint Space

## Train
- [ ] SmolVLA (LeRobot)

## Bench 환경 구성
- [ ] 기존 환경에 로봇팔 2개
  - [ ] 각자 단일 policy, 1개 로봇에만 VLSA 적용
    - [ ] VLSA Input (2개)
      - [ ] 반대편 로봇팔 EE의 Ellipsoid [Dynamic] (TF 변환) → 실시간
      - [ ] 장애물 [Static] (Grounding DINO)
    - [ ] 시나리오마다 객체 재배치 + 장애물 재배치

## Experiments (VLSA 점차 늘리기: 0 → 1 → 2)

### VLSA 아예 없는 버전
- [ ] bimanual test *(실패 예상)*
- [ ] bimanual + 장애물 test *(실패 예상)*

### 로봇팔 1개만 VLSA
- [ ] bimanual test *(성공)*
- [ ] bimanual + 장애물 test *(실패 예상)*

### 로봇팔 2개 다 VLSA
- [ ] bimanual test *(성공)*
- [ ] bimanual + 장애물 test *(성공)*

---

## 구현 메모
- 학습: 로봇 1개 (`preview_dual.py 1`) → single policy 학습
- 추론/벤치: 로봇 2개 (`preview_dual.py 2`) → 각자 단일 policy 실행, 한쪽에만 VLSA
- 환경: `LIBERO_Dual_Tabletop_Manipulation` (1~2 로봇 모두 지원)
