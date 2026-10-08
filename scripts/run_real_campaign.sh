#!/usr/bin/env bash
# Real multi-agent campaign (branch real-agents). Stages are resumable: compare.py --run only launches
# missing results, every GPU job goes through scripts/gpu_run.sh (one at a time).
#
#   bash scripts/run_real_campaign.sh <stage> [local|cloud]     (default: local)
#   HARDWARE=configs/hardware/box_48gb.yaml bash scripts/run_real_campaign.sh replay cloud   # GPU profile
#
#   bash scripts/run_real_campaign.sh tune_record   # vanilla live recordings on TUNING problems (train seed 101)
#   bash scripts/run_real_campaign.sh xcheck        # sequential replay of a tuning recording vs simulator
#   bash scripts/run_real_campaign.sh eval_record   # live vanilla @100 on test seeds 1-3 (= replay recordings)
#   bash scripts/run_real_campaign.sh replay        # HEADLINE: replay recordings, vanilla vs cachescout, 100/150/200
#   bash scripts/run_real_campaign.sh live          # remaining live runs (vanilla 150/200, cachescout 100/150/200)
#   LIVE_BLOCKS=100,150 bash scripts/run_real_campaign.sh live    # reduced live set if time is tight
#   bash scripts/run_real_campaign.sh gate_ablation # Fig. 14b: replay recordings, cachescout vs gate_off (needs replay)
#   bash scripts/run_real_campaign.sh rate_sweep    # Fig. 10b: live arrival-rate sweep, vanilla vs cachescout
#   RATES="0.2 0.5 1 2 4 8" SWEEP_SESSIONS=40 SWEEP_TOPO=selector SWEEP_SEED=1 SWEEP_BLOCKS=100 (defaults;
#   the rate list is fixed in docs/decisions.md before any run)
# Summaries of the last two: python scripts/summarize_box_ablations.py --exp real_cloud
set -u
cd "$(dirname "$0")/.."
stage=${1:?stage}
TARGET=${2:-local}
case "$TARGET" in
  local) CFG=configs/experiments/real/local.yaml; EXP=real ;;
  cloud) CFG=configs/experiments/real/cloud.yaml; EXP=real_cloud ;;
  *) echo "target must be local or cloud"; exit 2 ;;
esac
PY=${PYTHON:-python}
TOPOS="pipeline random debate selector"
SEEDS="1 2 3"
HW_ARGS=()
if [ -n "${HARDWARE:-}" ]; then HW_ARGS=(--hardware "$HARDWARE"); fi
echo "[campaign] stage=$stage target=$TARGET config=$CFG results=results/$EXP hardware=${HARDWARE:-config default}"

rec_dir() {  # rec_dir <split> <topology> <seed> <tag>
  echo "results/$EXP/gsm8k_$1_$2_s$3/vanilla/b100_$4"
}

case "$stage" in
  tune_record)
    for t in $TOPOS; do
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --systems vanilla --blocks 100 --tag rec --run \
        --set agents.topology=$t --set agents.split=train --set agents.problem_seed=101 || exit 1
    done ;;
  xcheck)
    REC=$(rec_dir train selector 101 rec)
    for s in vanilla lru_hook eviction_only; do
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --trace "$REC" --set mode=sequential --systems $s \
        --blocks 100 --tag seq --run || exit 1
      $PY scripts/crosscheck_sim_vs_gpu.py "results/$EXP/replay_gsm8k_train_selector_s101/$s/b100_seq/result.json" || exit 1
    done ;;
  eval_record)
    for seed in $SEEDS; do for t in $TOPOS; do
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --systems vanilla --blocks 100 --tag eval --run \
        --set agents.topology=$t --set agents.problem_seed=$seed || exit 1
    done; done ;;
  replay)
    for seed in $SEEDS; do for t in $TOPOS; do
      REC=$(rec_dir test $t $seed eval)
      [ -f "$REC/result.json" ] || { echo "missing recording $REC"; exit 1; }
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --trace "$REC" --set mode=online \
        --systems vanilla,cachescout --blocks 100,150,200 --tag replay --run || exit 1
    done; done ;;
  live)
    BL=${LIVE_BLOCKS:-100,150,200}
    for seed in $SEEDS; do for t in $TOPOS; do
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --systems vanilla,cachescout --blocks $BL --tag eval --run \
        --set agents.topology=$t --set agents.problem_seed=$seed || exit 1
    done; done ;;
  gate_ablation)   # cachescout results are shared with the replay stage (same tag), only gate_off runs
    for seed in $SEEDS; do for t in $TOPOS; do
      REC=$(rec_dir test $t $seed eval)
      [ -f "$REC/result.json" ] || { echo "missing recording $REC"; exit 1; }
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --trace "$REC" --set mode=online \
        --systems vanilla,cachescout,gate_off --blocks ${GATE_BLOCKS:-100,150,200} --tag replay --run || exit 1
    done; done ;;
  rate_sweep)      # live runs: each arrival rate gets its own tag; peak = max turns/s over rates
    for r in ${RATES:-0.2 0.5 1 2 4 8}; do
      $PY scripts/compare.py --config $CFG "${HW_ARGS[@]}" --systems vanilla,cachescout \
        --blocks ${SWEEP_BLOCKS:-100} --tag "rate$r" --run --set agents.topology=${SWEEP_TOPO:-selector} \
        --set agents.problem_seed=${SWEEP_SEED:-1} --set agents.arrival_rate=$r \
        --set agents.num_sessions=${SWEEP_SESSIONS:-40} || exit 1
    done ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
echo "stage $stage done"
