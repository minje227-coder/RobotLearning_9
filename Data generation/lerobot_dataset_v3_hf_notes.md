# Hugging Face LeRobotDataset v3.0 작업 메모

이 문서는 나중에 Hugging Face Hub 형식의 LeRobotDataset v3.0 데이터셋을 만들고, 그 데이터셋으로 학습할 때 Codex에게 그대로 넣어줄 컨텍스트용 메모다.

기준 문서:
- Hugging Face LeRobot docs `v0.5.1`
- 주제: `LeRobotDataset v3.0`

실행 전에 확인할 것:
- 현재 설치된 `lerobot` 버전
- 데이터셋 포맷이 여전히 v3.0인지
- 실제 사용할 로봇, 카메라 key, action/state shape
- Hugging Face repo id

## 목표

1. 로봇 학습용 시계열 데이터를 LeRobotDataset v3.0 형식으로 저장한다.
2. 데이터셋을 Hugging Face Hub에 push하거나 로컬에서 로드한다.
3. `LeRobotDataset` 또는 `StreamingLeRobotDataset`으로 학습에 사용한다.

## v3.0 핵심 변화

LeRobotDataset v3.0은 episode별 파일 저장 방식이 아니라 file-based shard 저장 방식을 사용한다.

- v2 방식: episode마다 Parquet/MP4 파일이 따로 존재
- v3 방식: 여러 episode가 하나의 Parquet/MP4 shard 안에 들어감
- episode 경계, 길이, offset, task 정보는 metadata에서 복원
- 파일 수가 줄어 대규모 데이터셋 초기화와 로딩이 더 안정적
- Hub에서 직접 streaming할 수 있음

## 디렉터리 구조

v3.0 데이터셋의 주요 구조:

```text
dataset_root/
  meta/
    info.json
    stats.json
    tasks.jsonl
    episodes/
      ...
  data/
    ...
  videos/
    ...
```

각 항목 의미:

- `meta/info.json`: feature schema, shape, dtype, FPS, codebase version, data/video shard path template
- `meta/stats.json`: normalization용 mean/std/min/max
- `meta/tasks.jsonl`: natural language task와 task id 매핑
- `meta/episodes/`: episode별 length, task, offset 정보
- `data/`: low-dimensional state/action/timestamp 등을 담는 Parquet shard
- `videos/`: camera별 MP4 shard

## 데이터 구성 원칙

Tabular data:
- `observation.state`
- `action`
- `timestamp`
- 기타 low-dimensional sensor 값
- Apache Parquet에 저장

Visual data:
- `observation.images.<camera_name>`
- camera frame을 MP4로 인코딩
- camera별 shard로 저장

Metadata:
- feature name
- dtype
- shape
- FPS
- normalization stats
- episode segmentation
- task id

중요한 설계 원칙:
- 파일 경계가 episode 경계가 아니다.
- episode별 접근은 metadata offset으로 복원한다.
- 코드에서 episode 파일명을 직접 가정하면 안 된다.

## recording 명령 예시

SO-101 기준 예시:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem585A0076841 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem58760431551 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/record-test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Grab the black cube" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

필요시 codec 자동 설정:

```bash
--dataset.vcodec=auto
```

## 직접 생성할 때 필수 규칙

Python API로 직접 dataset을 만들고 frame을 추가하는 경우, push 전 반드시 `finalize()`를 호출해야 한다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(...)

for episode in range(num_episodes):
    for frame in episode_data:
        dataset.add_frame(frame)
    dataset.save_episode()

dataset.finalize()
dataset.push_to_hub()
```

`finalize()`가 필요한 이유:

- buffered episode metadata를 disk에 flush
- Parquet writer를 닫아서 footer metadata 작성
- incomplete/corrupt Parquet 방지
- dataset load 실패 방지

push 전에 `finalize()`를 빼먹으면 v3.0 데이터셋이 깨질 수 있다.

## 학습용 로드 예시

Hub 또는 local cache에서 로드:

```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "HF_USER/DATASET_ID"

dataset = LeRobotDataset(repo_id)
sample = dataset[100]

print(sample.keys())
```

일반적으로 기대하는 key:

```text
observation.state
action
observation.images.front
timestamp
episode_index
frame_index
task_index
```

실제 key는 `meta/info.json`의 features를 기준으로 확인한다.

## temporal window 사용

특정 시점 기준 이전 frame들을 같이 넣고 싶으면 `delta_timestamps`를 사용한다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "HF_USER/DATASET_ID"

delta_timestamps = {
    "observation.images.front": [-0.2, -0.1, 0.0],
}

dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

sample = dataset[100]
print(sample["observation.images.front"].shape)
```

예상 shape:

```text
[T, C, H, W]
```

여기서 `T=3`.

## DataLoader 학습 루프 예시

```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "HF_USER/DATASET_ID"
dataset = LeRobotDataset(repo_id)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

for batch in data_loader:
    states = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.front"].to(device)

    # model.forward(batch)
```

## Hub streaming

대용량 데이터셋을 local disk에 전부 받지 않고 Hub에서 직접 streaming할 수 있다.

```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

repo_id = "HF_USER/DATASET_ID"
dataset = StreamingLeRobotDataset(repo_id)
```

사용 목적:

- 대용량 데이터셋 학습
- local storage 부족할 때
- Hub-native pipeline 테스트

주의:

- 네트워크 품질 영향을 받음
- training throughput이 local cached dataset보다 낮을 수 있음
- 장기 학습 전에는 throughput을 먼저 측정해야 함

## image transforms

LeRobotDataset은 학습 시점 image augmentation을 지원한다.

중요:
- recording 시 raw image를 저장한다.
- transform은 dataset 생성/recording 시점이 아니라 training load 시점에 적용한다.

기본 transform 예시:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig

transforms_config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=3,
    random_order=False,
)

dataset = LeRobotDataset(
    repo_id="HF_USER/DATASET_ID",
    image_transforms=ImageTransforms(transforms_config),
)
```

custom transform 예시:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransforms,
    ImageTransformsConfig,
    ImageTransformConfig,
)

custom_transforms_config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=2,
    random_order=True,
    tfs={
        "brightness": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"brightness": (0.7, 1.3)},
        ),
        "contrast": ImageTransformConfig(
            weight=2.0,
            type="ColorJitter",
            kwargs={"contrast": (0.8, 1.2)},
        ),
        "sharpness": ImageTransformConfig(
            weight=0.5,
            type="SharpnessJitter",
            kwargs={"sharpness": (0.3, 2.0)},
        ),
    },
)

dataset = LeRobotDataset(
    repo_id="HF_USER/DATASET_ID",
    image_transforms=ImageTransforms(custom_transforms_config),
)
```

torchvision transform도 직접 넣을 수 있다.

```python
from torchvision.transforms import v2
from lerobot.datasets.lerobot_dataset import LeRobotDataset

torchvision_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

dataset = LeRobotDataset(
    repo_id="HF_USER/DATASET_ID",
    image_transforms=torchvision_transforms,
)
```

지원 transform:

- `ColorJitter`: brightness, contrast, saturation, hue
- `SharpnessJitter`: image sharpness
- `Identity`: no-op
- 그 외 `torchvision.transforms.v2` transform

권장:

- 처음에는 약한 augmentation부터 시작
- 예: brightness `0.9-1.1`, contrast `0.9-1.1`
- 실제 robot domain의 lighting variation에 맞게 조절
- 너무 강한 augmentation은 policy 성능을 떨어뜨릴 수 있음

시각화:

```bash
lerobot-imgtransform-viz \
  --repo-id=HF_USER/DATASET_ID \
  --output-dir=./transform_examples \
  --n-examples=5
```

## v2.1에서 v3.0으로 변환

pre-release v3 지원 build 설치 예시:

```bash
pip install "https://github.com/huggingface/lerobot/archive/33cad37054c2b594ceba57463e8f11ee374fa93c.zip"
```

Hub에 있는 기존 v2.1 dataset 변환:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=HF_USER/DATASET_ID
```

변환 과정:

- episode별 Parquet을 shard Parquet으로 aggregate
- episode별 MP4를 shard MP4로 aggregate
- `meta/episodes/*`에 episode length, task, offset 기록

## 데이터셋 만들 때 체크리스트

데이터 정의:

- `repo_id`: `HF_USER/DATASET_ID`
- `fps`
- `task` 또는 task list
- `observation.state` shape
- `action` shape
- camera names
- image resolution
- episode count
- episode당 frame 수

파일/메타데이터 확인:

- `meta/info.json` 존재
- `meta/stats.json` 존재
- `meta/tasks.jsonl` 존재
- `meta/episodes/` 존재
- `data/` Parquet shard 존재
- `videos/` MP4 shard 존재
- `LeRobotDataset(repo_id_or_path)`로 load 가능
- random index access 가능
- `DataLoader` batch 생성 가능

학습 전 확인:

- `sample.keys()`
- state/action dtype
- state/action shape
- image tensor shape: `[C, H, W]` 또는 temporal window 사용 시 `[T, C, H, W]`
- timestamp 단위
- task conditioning 필요 여부
- normalization stats 적용 방식
- train/val split 방식

## 자주 생기는 문제

`finalize()` 누락:
- Parquet footer가 없어서 dataset load 실패 가능
- push 전에 반드시 호출

camera key 불일치:
- 예: 코드에서는 `observation.images.front`를 기대하지만 dataset은 `observation.images.front_left`
- 항상 `sample.keys()`와 `meta/info.json` 확인

episode 파일명 가정:
- v3.0에서는 episode별 파일이 아닐 수 있음
- `episode-0000.parquet` 같은 이름에 의존하지 말 것

streaming throughput:
- Hub streaming은 편하지만 네트워크 영향을 받음
- 긴 학습은 local cache와 비교 필요

augmentation 과함:
- image transform이 너무 강하면 imitation learning 성능 저하 가능
- 먼저 visualization script로 확인

## 나중에 Codex에게 줄 요청 예시

```text
이 문서를 기준으로 LeRobotDataset v3.0 데이터셋 생성 파이프라인을 만들어줘.
현재 데이터는 <데이터 경로>에 있고, 목표 repo_id는 <HF_USER/DATASET_ID>야.
state/action/image key를 먼저 확인한 뒤 v3.0 형식으로 저장하고,
LeRobotDataset으로 로드되는지 검증하고,
학습 스크립트에서 DataLoader까지 연결해줘.
push_to_hub 전에는 반드시 finalize() 호출 여부를 확인해줘.
```

## 나중에 필요한 입력값

작업을 시작하려면 최소한 아래 값을 정해야 한다.

```text
HF_USER=
DATASET_ID=
LOCAL_RAW_DATA_PATH=
LOCAL_OUTPUT_DATASET_PATH=
FPS=
TASK_DESCRIPTION=
STATE_KEY=
ACTION_KEY=
CAMERA_KEYS=
IMAGE_SIZE=
NUM_EPISODES=
TRAINING_POLICY=
```

