#!/usr/bin/env bash
# VLSA {0,1,2} x delay {7.5..12} x seed {0..9} grid. VLSA-1 = safety on robot0.
# Metrics only (no video, no ETS). ckpt 015000.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/workspace/RobotLearning_9/Test || exit 1

P0=$HOME/workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model
P1=$HOME/workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model
DELAYS="7.5 8 8.5 9 9.5 10 10.5 11 11.5 12"
SEEDS="0 1 2 3 4 5 6 7 8 9"
mkdir -p results/grid

runjob() {  # $1=gpu $2=vlsa $3=delay
  local gpu=$1 vl=$2 d=$3 sa=""
  [ "$vl" = "1" ] && sa="--safety-arms 0"
  [ "$vl" = "2" ] && sa="--safety-arms 0 1"
  local lab="grid_VLSA${vl}_d${d//./p}"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python minje_run_safety_eval.py \
    --device cuda --seeds $SEEDS \
    --robot0-policy-path "$P0" --robot1-policy-path "$P1" \
    --robot0-task "put the milk into the black box" \
    --robot1-task "put the orange juice into the black box" \
    --max-steps 500 --robot1-start-delay-sec "$d" $sa \
    --policy-replan-horizon 50 --trajectory-horizon 0 \
    --out-csv "results/grid/${lab}.csv" --out-json "results/grid/${lab}.json" \
    --label "$lab" > "results/grid/${lab}.log" 2>&1
}

# 30 jobs (3 vlsa x 10 delays); distribute round-robin over 6 GPUs (0-5), sequential per GPU.
jobs=()
for vl in 0 1 2; do for d in $DELAYS; do jobs+=("$vl:$d"); done; done
for g in 0 1 2 3 4 5; do
  (
    idx=$g
    while [ $idx -lt ${#jobs[@]} ]; do
      job=${jobs[$idx]}
      runjob $g "${job%%:*}" "${job##*:}"
      idx=$((idx + 6))
    done
  ) &
done
wait
echo GRID_DONE
