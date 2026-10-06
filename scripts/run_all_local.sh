#!/usr/bin/env bash
# Full local M5 campaign (RTX 4060 8 GB). Every GPU job goes through scripts/gpu_run.sh (one at a time,
# timeout, memory monitor). Results that already exist are reused (compare.py only runs missing ones).
# Total runtime when nothing exists yet: roughly 2-2.5 hours.
#
#   source ~/testLLM/.venv/bin/activate && bash scripts/run_all_local.sh
set -u
cd "$(dirname "$0")/.."
CFG=configs/experiments/main/local.yaml
PY=${PYTHON:-python}

step() { echo; echo "=== $(date +%T) $* ==="; }

step "1/5 cache-size sweep (Fig. 14) + baselines (Fig. 8) + mechanism ablation (Fig. 12)"
$PY scripts/compare.py --config $CFG --blocks 100,150,200 \
  --systems vanilla,continuum,warmup_only,eviction_only,cachescout --run || exit 1

step "2/5 extra ablations: literal Alg. 1, no prediction (tau = 0)"
$PY scripts/compare.py --config $CFG --blocks 100,150,200 \
  --systems vanilla,eviction_only,cachescout,cachescout_literal,no_prediction --run || exit 1

step "3/5 seed variation (evaluation seeds 2, 3)"
for t in selector_eval_s2 selector_eval_s3; do
  $PY scripts/compare.py --config $CFG --trace results/traces/$t.json --blocks 100,150,200 \
    --systems vanilla,eviction_only,cachescout --run || exit 1
done

step "4/5 coordination topologies (Fig. 4/5): pipeline, debate, random"
for t in pipeline_eval debate_eval random_eval; do
  $PY scripts/compare.py --config $CFG --trace results/traces/$t.json --blocks 100,150 \
    --systems vanilla,cachescout --run || exit 1
done

step "5/5 load sweep (Fig. 11-style): 0.5 / 1 / 2 / 4 sessions/s at 150 blocks"
for t in selector_eval selector_eval_r1 selector_eval_r2 selector_eval_r4; do
  $PY scripts/compare.py --config $CFG --trace results/traces/$t.json --blocks 150 \
    --systems vanilla,cachescout --run || exit 1
done

step "done"
