#!/usr/bin/env bash
# VLSA {0,1,2} x delay {8,12} matrix, minje harness, ckpt 015000, 50 seeds,
# video (normal + ellipsoid overlay) for every seed.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/workspace/RobotLearning_9/Test || exit 1

P0=$HOME/workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model
P1=$HOME/workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model
OUT=$HOME/workspace/RobotLearning_9/Test/minje_safety_test/outputs
SEEDS=$(seq 1 50)

run() {  # $1=gpu $2=label $3=delay ; remaining args = extra (e.g. --safety-arms ...)
  local gpu=$1 label=$2 delay=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python minje_run_safety_eval.py \
    --device cuda --seed-start 1 --num-episodes 50 \
    --robot0-policy-path "$P0" --robot1-policy-path "$P1" \
    --robot0-task "put the milk into the black box" \
    --robot1-task "put the orange juice into the black box" \
    --max-steps 500 --robot1-start-delay-sec "$delay" \
    --video-camera all --video-seeds $SEEDS --video-dir "$OUT" \
    "$@" \
    --policy-replan-horizon 50 --trajectory-horizon 0 \
    --out-csv "results/minje_${label}.csv" --out-json "results/minje_${label}.json" \
    --label "$label" > "results/minje_${label}.log" 2>&1
}

# 6 free GPUs (0-5), one run each (no chaining) — re-render with filled ellipsoids.
( run 0 VLSA0_d8  8 ) &
( run 1 VLSA0_d12 12 ) &
( run 2 VLSA1_d8  8  --safety-arms 1 ) &
( run 3 VLSA1_d12 12 --safety-arms 1 ) &
( run 4 VLSA2_d8  8  --safety-arms 0 1 ) &
( run 5 VLSA2_d12 12 --safety-arms 0 1 ) &
wait
echo ALL_DONE
