# dual panda side+wrist LeRobotDataset v3 TODO

목표: `robot0`용 scripted rollout과 `robot1`용 scripted rollout을 각각 raw episode로 모으고, 나중에 합쳐서 SmolVLA 학습용 LeRobotDataset v3로 쓴다.

현재 기준 스크립트:

- `robot0`: [create_dataset.py](/home/minje/RobotLearning_9/Data generation/create_dataset.py)
- `robot1`: [create_dataset_robot1.py](/home/minje/RobotLearning_9/Data generation/create_dataset_robot1.py)
- `robot1` 디버그 기준: [auto_scene2.py](/home/minje/RobotLearning_9/Data generation/auto_scene2.py)
- 변환 스크립트: [convert_lerobot_dataset.py](/home/minje/RobotLearning_9/Data generation/convert_lerobot_dataset.py)

기본 방향:

- `agentview`는 쓰지 않는다.
- 학습 view는 `side + wrist`만 쓴다.
- `birdview`, `backview`는 디버그 비디오용으로만 쓴다.
- 실패 episode는 저장하지 않는다.
- 최종 학습은 `robot0` raw와 `robot1` raw를 합친 데이터셋으로 진행한다.

## 1. 현재 확정 사항

view:

- `sideview_image` -> `observation.images.side`
- `robot0_eye_in_hand_image` 또는 `robot1_eye_in_hand_image` -> `observation.images.wrist`

state / action:

- `observation.state`: 8D
- `action`: 8D
- 구성: `7 joint position or delta + 1 gripper scalar/cmd`

이미지 처리:

- 현재 가정은 `vertical_flip`
- 저장 전 `img = obs[key][::-1]`

성공 판정:

- 성공 episode만 raw `.npz` 저장
- 실패 episode는 JSONL log만 남김

랜덤 배치:

- `milk_1`, `milk_2`, `trash target` 모두 seed 기반 랜덤 샘플링
- 같은 seed면 같은 배치 재현 가능

## 2. robot0 / robot1 역할 분리

`robot0` dataset:

- active arm: `robot0`
- target object: `milk_1`
- wrist view: `robot0_eye_in_hand`
- 기준 디버그 스크립트: `auto_scene.py`

`robot1` dataset:

- active arm: `robot1`
- target object: `milk_2`
- wrist view: `robot1_eye_in_hand`
- 기준 디버그 스크립트: `auto_scene2.py`

공통:

- passive arm은 workspace 밖 home pose로 고정
- dual action으로 env step
- dataset에는 active arm의 8D action만 저장

## 3. robot1 쪽 현재 반영 내용

- [x] `auto_scene2.py` 기준 orientation 복구
- [x] `TARGET_FORWARD_AXIS_WORLD = [1.0, 0.0, 0.0]`
- [x] `lift -> preplace_mid -> preplace -> place` waypoint 흐름 반영
- [x] `move_preplace_mid` 추가
- [x] `JOINT_TOL_MID = 0.4`
- [x] `move_preplace` tolerance override 유지
- [x] `interpolate_joint_waypoint()` 추가

tolerance 기준:

- 일반 waypoint: `JOINT_TOL = 0.15`
- `_mid`: `JOINT_TOL_MID = 0.4`
- `move_preplace`: `0.25`

## 4. raw episode 저장 형식

현재 raw `.npz` key:

- `side`
- `wrist`
- `state`
- `action`
- `task`

shape:

- `side`: `[T, H, W, 3]`
- `wrist`: `[T, H, W, 3]`
- `state`: `[T, 8]`
- `action`: `[T, 8]`

## 5. LeRobotDataset v3 feature 정의

```python
features = {
    "observation.images.side": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["action"],
    },
}
```

## 6. 변환 스크립트 상태

- [x] raw `.npz` 구조와 `convert_lerobot_dataset.py`는 호환됨
- [ ] `robot_type`만 최종 기준으로 확정 필요

메모:

- `robot0`용 변환이면 `robot_type="dual_panda_robot0"`
- `robot1`용 변환이면 `robot_type="dual_panda_robot1"`
- 최종 병합 데이터셋을 어떻게 표기할지는 나중에 결정

## 7. 최종 병합 계획

목표:

- `robot0` raw dataset 수집
- `robot1` raw dataset 수집
- 둘을 하나의 학습 dataset으로 합치기

후보 방식:

- 방법 A: raw `.npz`를 한 폴더에 모아 한 번에 LeRobot v3로 변환
- 방법 B: `robot0` v3, `robot1` v3를 따로 만든 뒤 병합용 스크립트 작성

현재 권장:

- 방법 A 우선 검토
- 이유: `task`/metadata/statistics를 한 번에 다시 쓰는 편이 덜 꼬임

## 8. description / task 문구

현재:

- raw 저장 시 `TASK_DESCRIPTION` 문자열이 그대로 `task`로 들어감

해야 할 일:

- [ ] 최종 `description/task` 문구 확정
- [ ] `robot0`, `robot1`를 같은 문구로 넣을지 따로 넣을지 결정
- [ ] 병합 dataset에서 task 문구를 통일할지 결정

메모:

- 이건 raw 생성 전에도 바꿀 수 있고
- raw가 이미 있으면 변환 단계에서 덮어쓸 수도 있음

## 9. 검증 TODO

- [x] `robot0` 수집 스크립트 작성
- [x] `robot1` 수집 스크립트 작성
- [x] `robot1` scripted path를 `auto_scene2.py` 기준으로 반영
- [x] state/action 8D 확정
- [x] side+wrist view 확정
- [x] 실패 episode 미저장 확정
- [ ] `robot0` 5 episode 테스트
- [ ] `robot1` 5 episode 테스트
- [ ] `robot0` raw -> LeRobot v3 변환 확인
- [ ] `robot1` raw -> LeRobot v3 변환 확인
- [ ] 병합 방식 결정
- [ ] 병합 dataset 1회 생성 테스트
- [ ] `meta/tasks.jsonl` 확인
- [ ] `dataset[0]` sample 확인
- [ ] `sample["observation.state"].shape == (8,)`
- [ ] `sample["action"].shape == (8,)`
- [ ] side/wrist 이미지 방향 확인
- [ ] SmolVLA 100 step 짧은 학습 테스트

## 10. 실행 메모

raw 수집:

- `libero` env에서 실행
- LeRobot writer는 현재 `libero` env에서 바로 안 되므로 raw `.npz` 우선 저장

변환:

- `lerobot` env에서 실행

주의:

- `--push-to-hub false`처럼 쓰지 말고, false면 플래그 자체를 빼기
- path에 공백이 있으니 `"/home/minje/RobotLearning_9/Data generation/..."`처럼 감싸기
- `trash_can.xml` 공유 asset 때문에 병렬 worker는 충돌 가능성 있음

## 11. 현재 가정

```text
SAVE_FAILURE_EPISODES=false
IMAGE_ROTATION=vertical_flip
FPS=10
RESOLUTION=256
IMAGE_FEATURES=side,wrist
STATE_DIM=8
ACTION_DIM=8
FINAL_TRAINING_DATASET=robot0 + robot1 merged
```
