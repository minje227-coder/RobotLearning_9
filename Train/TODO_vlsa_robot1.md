# SmolVLA Training TODO - Whalswp/vlsa_robot1

목표: `/home/minje/RobotLearning_9/Train/lerobot` repo에서 Hugging Face dataset `Whalswp/vlsa_robot1`로 SmolVLA를 학습한다.

기준:

- 작업 repo: `/home/minje/RobotLearning_9/Train/lerobot`
- 학습 스크립트: `src/lerobot/scripts/lerobot_train.py`
- policy: `smolvla`
- dataset repo id: `Whalswp/vlsa_robot1`
- dataset revision:
  - `Whalswp/vlsa_robot0`: `d90fe30e10765b08ee8bf29f5a2aa1cdc901c768`
  - `Whalswp/vlsa_robot1`: `559b5ab1ce2cc44e344873d2e98238454b99a904`
- 예상 feature:
  - `observation.images.side`
  - `observation.images.wrist`
  - `observation.state`: 8D
  - `action`: 8D

## 0. 결정된 값 / 나중에 줄 인수

- [x] base policy path: `lerobot/smolvla_base`
- [x] `policy.train_expert_only=false`
  - 실행 명령에 인수로 명시해서 줄 것
- [ ] batch size: 나중에 실행할 때 인수로 지정
  - smoke test 기본값만 `8`
- [ ] 본 학습 steps: 나중에 실행할 때 인수로 지정
- [x] wandb 사용
  - `--wandb.enable=true`
  - project 기본값: `smolvla`
- [x] dataset repo에 version tag가 없으므로 `--dataset.revision`을 반드시 명시
- [x] `smolvla_base`는 camera 이름을 `camera1`, `camera2`, `camera3`로 기대함
  - dataset `side` -> policy `camera1`
  - dataset `wrist` -> policy `camera2`
  - 없는 `camera3`는 `--policy.empty_cameras=1`로 빈 카메라 처리

## 1. 환경 확인

repo root에서 실행:

```bash
cd /home/minje/RobotLearning_9/Train/lerobot
```

- [ ] 현재 변경 상태 확인

```bash
git status --short
```

- [ ] Python / CUDA / torch 확인

```bash
python - << "PY"
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
PY
```

- [ ] LeRobot import 확인

```bash
python - << "PY"
import lerobot
print("lerobot", getattr(lerobot, "__version__", "unknown"))
PY
```

- [ ] SmolVLA config 확인

```bash
python - << "PY"
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
cfg = SmolVLAConfig()
print("type", cfg.type)
print("chunk_size", cfg.chunk_size)
print("n_action_steps", cfg.n_action_steps)
print("max_state_dim", cfg.max_state_dim)
print("max_action_dim", cfg.max_action_dim)
print("train_expert_only", cfg.train_expert_only)
PY
```

## 2. Dataset 로드 검증

- [ ] `Whalswp/vlsa_robot1`이 현재 환경에서 로드되는지 확인

```bash
python - << "PY"
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "Whalswp/vlsa_robot1"
revision = "559b5ab1ce2cc44e344873d2e98238454b99a904"
ds = LeRobotDataset(repo_id, revision=revision)
print("len", len(ds))
print("features", list(ds.meta.info["features"].keys()))

sample = ds[0]
print("sample keys", list(sample.keys()))
for key in [
    "observation.images.side",
    "observation.images.wrist",
    "observation.state",
    "action",
]:
    v = sample[key]
    print(key, getattr(v, "shape", None), getattr(v, "dtype", None))
PY
```

- [ ] 확인할 것
  - `len(ds) > 0`
  - `observation.images.side` 존재
  - `observation.images.wrist` 존재
  - `observation.state` shape가 8D
  - `action` shape가 8D
  - `fps == 10`인지 `meta/info.json`에서 확인

- [ ] 이미지 방향 확인
  - side/wrist sample frame을 저장해서 upside-down 여부 확인
  - 뒤집혀 있으면 dataset 생성 쪽에서 이미 `vertical_flip`이 들어갔는지 다시 확인

## 3. 100 step smoke test

목표: dataloader, SmolVLA forward/backward, checkpoint 저장까지 먼저 확인한다.

주의:

- `output_dir`가 이미 있으면 `resume=false` 상태에서 에러가 난다.
- 다시 돌릴 때는 output dir 이름을 바꾸거나 resume을 사용한다.

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=Whalswp/vlsa_robot1 \
  --dataset.revision=559b5ab1ce2cc44e344873d2e98238454b99a904 \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size=8 \
  --steps=100 \
  --save_freq=100 \
  --log_freq=10 \
  --eval_freq=-1 \
  --num_workers=4 \
  --output_dir=./outputs/smolvla_vlsa_robot1_smoke \
  --job_name=smolvla_vlsa_robot1_smoke \
  --wandb.enable=true \
  --wandb.project=smolvla \
  --wandb.disable_artifact=true
```

- [ ] smoke test 성공 기준
  - dataset load 에러 없음
  - feature key mismatch 없음
  - state/action shape mismatch 없음
  - CUDA OOM 없음
  - loss가 NaN이 아님
  - checkpoint 생성됨

예상 checkpoint:

```text
./outputs/smolvla_vlsa_robot1_smoke/checkpoints/000100/pretrained_model
```

## 4. 본 학습

smoke test가 통과하면 실행한다. `{BATCH_SIZE}`와 `{STEPS}`는 실행할 때 정해서 넣는다.

```bash
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=Whalswp/vlsa_robot1 \
  --dataset.revision=559b5ab1ce2cc44e344873d2e98238454b99a904 \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.train_expert_only=false \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size={BATCH_SIZE} \
  --steps={STEPS} \
  --save_freq=5000 \
  --log_freq=50 \
  --eval_freq=-1 \
  --num_workers=8 \
  --output_dir=./outputs/smolvla_vlsa_robot1 \
  --job_name=smolvla_vlsa_robot1 \
  --wandb.enable=true \
  --wandb.project=smolvla \
  --wandb.disable_artifact=true
```

- [ ] 첫 100~500 step에서 loss 정상 확인
- [ ] GPU memory 확인
- [ ] OOM이면 `batch_size=32`, 그래도 안 되면 `16`
- [ ] checkpoint가 `5000` step마다 저장되는지 확인
- [ ] 최종 checkpoint 경로 기록

## 5. Resume

중단되면 `train_config.json` 기준으로 resume한다.

먼저 실제 config 위치 확인:

```bash
find ./outputs/smolvla_vlsa_robot1/checkpoints -name train_config.json -print
```

resume 예시:

```bash
python src/lerobot/scripts/lerobot_train.py \
  --config_path=./outputs/smolvla_vlsa_robot1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

- [ ] step이 0부터 재시작하지 않는지 확인
- [ ] 기존 output dir에 이어서 checkpoint가 생기는지 확인

## 6. 학습 후 확인

- [ ] checkpoint 목록 확인

```bash
find ./outputs/smolvla_vlsa_robot1/checkpoints -maxdepth 3 -type f -name train_config.json -print
```

- [ ] 최종 사용할 checkpoint 기록
  - 후보: `./outputs/smolvla_vlsa_robot1/checkpoints/last/pretrained_model`
  - 후보: `./outputs/smolvla_vlsa_robot1/checkpoints/030000/pretrained_model`

- [ ] checkpoint 안에 필요한 파일 확인
  - `config.json`
  - `train_config.json`
  - tokenizer/processor 파일
  - model weight 파일

## 7. 문제가 나면 먼저 볼 것

- dataset key mismatch
  - dataset key가 `observation.images.side`, `observation.images.wrist`인지 확인
  - 필요하면 `rename_map` 사용 가능 여부 확인
- state/action shape mismatch
  - dataset은 8D, SmolVLA config는 `max_state_dim=32`, `max_action_dim=32`
  - padding이 적용되는지 확인
- CUDA OOM
  - batch size 낮추기
  - `num_workers` 낮추기
- Hub 접근 실패
  - `huggingface-cli whoami`
  - private dataset이면 token/login 확인
- output dir exists
  - 새 output dir 사용 또는 resume 사용

## 8. 기록할 값

- [ ] dataset revision/hash
- [ ] base policy repo/path
- [ ] `train_expert_only`
- [ ] batch size
- [ ] steps
- [ ] learning rate
- [ ] GPU 종류/개수
- [ ] smoke test 결과
- [ ] 최종 checkpoint path
