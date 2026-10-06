---
description: Record an experiment's config, results, and notes in results/log/
argument-hint: <experiment name or path to result dir> [notes]
---

Log the experiment **$ARGUMENTS**.

1. Locate the run's resolved config and metrics (in `results/` or the path given). If missing, ask; never fabricate numbers.
2. Collect provenance: `git rev-parse --short HEAD` (+ whether the tree is dirty), date (`date -Iseconds`),
   seed(s), hardware profile, and versions via
   `~/testLLM/.venv/bin/python -c "import vllm,torch,transformers,sys;print(vllm.__version__,torch.__version__,transformers.__version__,sys.version.split()[0])"`.
3. Write `results/log/<YYYY-MM-DD>_<experiment>.md` containing:
   - Purpose and milestone (docs/PLAN.md)
   - Full resolved config (YAML block)
   - Provenance (commit, versions, GPU, seeds)
   - Results table (hit rate, mean/median/P99 TTFT, per-turn latency, throughput) next to the baseline
   - Comparison with the paper's corresponding numbers (cite figure), noting the scale difference
   - Notes / anomalies / next steps (include any notes passed in the arguments)
4. Also write the same data as `results/log/<YYYY-MM-DD>_<experiment>.json`.
5. Don't overwrite an existing log file. Add a suffix `_2`, `_3`, and so on.
