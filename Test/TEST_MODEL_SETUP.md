# test_model.py 설치/실행 가이드

## 1) 권장 환경

- Python: `3.12` (필수)
- conda env 예시 이름: `lerobot_eval`

```bash
conda create -n lerobot_eval python=3.12 -y
conda activate lerobot_eval
```

## 2) 필수 패키지 설치

```bash
# 이 저장소 루트로 이동
cd /path/to/RobotLearning_9

# 기본
pip install --upgrade pip
pip install torch torchvision
pip install imageio pandas huggingface_hub draccus safetensors

# LIBERO 런타임 호환
pip install robosuite==1.4.1
pip install bddl easydict
```

## 3) LeRobot 소스 경로 사용

`test_model.py`는 아래 경로를 프로젝트 루트 기준으로 참조합니다.

- `Train/lerobot/src`
- `Data generation`

별도 `pip install -e Train/lerobot` 없이도 실행 가능하도록 되어 있습니다.

## 4) 실행 예시

### 4-1. 로컬 경로 사용

```bash
cd /path/to/RobotLearning_9/Test

python test_model.py \
  --robot0-policy-path "../outputs/my_smolvla/checkpoints/100000/pretrained_model" \
  --robot1-policy-path "../outputs/my_smolvla/checkpoints/100000/pretrained_model" \
  --seed 1 \
  --max-steps 400 \
  --video-out "./test_policy_episode.mp4"
```

### 4-2. 환경변수 방식(로컬 경로)

```bash
export PROJECT_ROOT=/path/to/RobotLearning_9
cd "$PROJECT_ROOT/Test"

python test_model.py \
  --robot0-policy-path "$PROJECT_ROOT/outputs/my_smolvla/checkpoints/100000/pretrained_model" \
  --robot1-policy-path "$PROJECT_ROOT/outputs/my_smolvla/checkpoints/100000/pretrained_model" \
  --seed 1 \
  --max-steps 400 \
  --video-out "$PROJECT_ROOT/Test/test_policy_episode.mp4"
```

### 4-3. HF repo id 사용 (dataset + policy 모두)

```bash
cd /path/to/RobotLearning_9/Test

python test_model.py \
  --robot0-policy-path "your_org/your_robot0_policy_repo" \
  --robot1-policy-path "your_org/your_robot1_policy_repo" \
  --seed 1 \
  --max-steps 400 \
  --video-out "./test_policy_episode.mp4"
```

## 5) HF 모델 경로 사용

- 로컬 경로 대신 HF repo id 사용 가능
  - 예: `--robot0-policy-path yourname/your-policy-repo`

## 6) 자주 나는 에러

- `TypeError: 'type' object is not subscriptable`
  - Python 3.8/3.9 실행 오류. Python 3.12로 실행 필요.

- `SyntaxError` at `def ...[T: ...]`
  - Python 3.10 이하 실행 오류. Python 3.12로 실행 필요.

- `ModuleNotFoundError: safetensors/bddl/easydict`
  - 위 2) 패키지 설치 누락.

- `CUDA out of memory`
  - 두 policy 동시 로딩 시 발생 가능.
  - 우선 `--device cpu`로 확인 후, GPU 메모리 여유 확보 후 재실행 권장.
