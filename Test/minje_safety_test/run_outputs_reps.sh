#!/usr/bin/env bash
# Re-render one representative per (VLSA, delay8/12, case-category) for outputs/,
# with the fixed per-arm/per-robot intervene colors. Matches outputs matrix config
# (VLSA-1 = robot1-safety). Keep overlay only; delete the old 600 afterwards.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/workspace/RobotLearning_9/Test || exit 1
P0=$HOME/workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model
P1=$HOME/workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model
OUT=$HOME/workspace/RobotLearning_9/Test/minje_safety_test/outputs

# "vl delay seed seed ..."   (safety set by vl inside render)
JOBS=(
"0 8 1"
"0 12 17 1"
"1 8 1"
"1 12 10 1"
"2 8 1 7 2 19"
"2 12 9 16 1"
)

render() {  # $1=gpu  $2="vl dl s..."
  local gpu=$1; local arr=($2); local vl=${arr[0]} dl=${arr[1]}; local seeds="${arr[@]:2}"
  local sa=""
  [ "$vl" = "1" ] && sa="--safety-arms 1"
  [ "$vl" = "2" ] && sa="--safety-arms 0 1"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl python minje_run_safety_eval.py \
    --device cuda --seeds $seeds \
    --robot0-policy-path "$P0" --robot1-policy-path "$P1" \
    --robot0-task "put the milk into the black box" \
    --robot1-task "put the orange juice into the black box" \
    --max-steps 500 --robot1-start-delay-sec "$dl" $sa \
    --policy-replan-horizon 50 --trajectory-horizon 0 \
    --video-camera all --video-seeds $seeds --video-dir "$OUT" \
    --out-csv "/tmp/rep_V${vl}_d${dl}.csv" --out-json "/tmp/rep_V${vl}_d${dl}.json" \
    --label "REP_V${vl}_d${dl}" > "$OUT/rep_render_V${vl}_d${dl}.log" 2>&1
}

i=0
for job in "${JOBS[@]}"; do
  render $((i % 6)) "$job" &
  i=$((i + 1))
done
wait
echo REPS_DONE
