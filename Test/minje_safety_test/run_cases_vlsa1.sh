#!/usr/bin/env bash
# Render one VLSA-1 video per (delay, case-category). Filled+intervene overlay.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/workspace/RobotLearning_9/Test || exit 1
P0=$HOME/workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model
P1=$HOME/workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model
OUT=$HOME/workspace/RobotLearning_9/Test/minje_safety_test/cases_vlsa1
mkdir -p "$OUT"

# "delaylabel delayval seed seed ..."
JOBS=(
"7p5 7.5 0 1"
"8 8 0 1"
"8p5 8.5 0 6"
"9 9 0 7 9"
"9p5 9.5 0 1 2"
"10 10 0 1 2"
"10p5 10.5 0 1 2"
"11 11 0 1 2"
"11p5 11.5 0 1 2"
"12 12 0 1 2"
)

render() {  # $1=gpu  $2="dlabel dval s..."
  local gpu=$1; local arr=($2); local dl=${arr[0]} dv=${arr[1]}; local seeds="${arr[@]:2}"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python minje_run_safety_eval.py \
    --device cuda --seeds $seeds \
    --robot0-policy-path "$P0" --robot1-policy-path "$P1" \
    --robot0-task "put the milk into the black box" \
    --robot1-task "put the orange juice into the black box" \
    --max-steps 500 --robot1-start-delay-sec "$dv" --safety-arms 0 \
    --policy-replan-horizon 50 --trajectory-horizon 0 \
    --video-camera all --video-seeds $seeds --video-dir "$OUT" \
    --out-csv "/tmp/case_v1_d${dl}.csv" --out-json "/tmp/case_v1_d${dl}.json" \
    --label "CASE_V1_d${dl}" > "$OUT/render_d${dl}.log" 2>&1
}

i=0
for job in "${JOBS[@]}"; do
  render $((i % 6)) "$job" &
  i=$((i + 1))
  [ $((i % 6)) -eq 0 ] && wait
done
wait
echo RENDER_DONE
