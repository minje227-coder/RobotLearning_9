#!/usr/bin/env bash
# Remaining margins 0.04,0.06,0.08,0.10. VLSA-1 (robot0 safety), delay=10, seeds 0-49.
# Fixes the run_margin2.sh bug: `lab` is assigned on its own line so $m is set first.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/workspace/RobotLearning_9/Test || exit 1
P0=$HOME/workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model
P1=$HOME/workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model
SEEDS="$(seq 0 49)"
mkdir -p results/margin
runjob() {
  local gpu=$1 m=$2
  local lab="margin_${m//./p}"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python minje_run_safety_eval.py \
    --device cuda --seeds $SEEDS --robot0-policy-path "$P0" --robot1-policy-path "$P1" \
    --robot0-task "put the milk into the black box" --robot1-task "put the orange juice into the black box" \
    --max-steps 500 --robot1-start-delay-sec 10 --safety-arms 0 \
    --cbf-margin "$m" --policy-replan-horizon 50 --trajectory-horizon 0 \
    --out-csv "results/margin/${lab}.csv" --out-json "results/margin/${lab}.json" \
    --label "$lab" > "results/margin/${lab}.log" 2>&1
}
( runjob 0 0.04 ; runjob 0 0.08 ) &
( runjob 2 0.06 ; runjob 2 0.10 ) &
wait
echo MARGIN3_DONE
