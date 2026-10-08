# Cloud / remote-box runbook (M6): run the replication on a 24, 48 or 80 GB GPU

Everything here is for **you** to run on a remote Linux machine; nothing in this repo accesses any cloud account.
**First do `docs/NEW_BOX_SETUP.md`** (connect, clone, Python, vLLM, Hugging Face, preflight). The one-page
version is `docs/BOX_CHEATSHEET.md`.

**What changes vs the laptop:**
- model Llama-3.1-8B-Instruct, the paper's main model (Sec. 5.1);
- a GPU-size profile (`configs/hardware/box_24gb.yaml`, `box_48gb.yaml`, `box_80gb.yaml`), chosen with
  `--hardware` (scripts) or `HARDWARE=` (campaign);
- traces with 300 sessions and arrival rates up to 50 sessions/s (`configs/traces/cloud.yaml`, Llama token range);
- cloud experiment configs `configs/experiments/*/cloud.yaml`.

Algorithm constants are **identical** to the laptop's (tuned on tuning traces; `docs/decisions.md`).
`configs/hardware/cloud.yaml` is the original 80 GB profile; `box_80gb.yaml` has the same vLLM settings but
offline mode.

Which experiments fit which GPU size, with the memory arithmetic: `docs/NEW_BOX_SETUP.md` §9. Short version:
everything fits on 48 and 80 GB. On 24 GB, everything at the paper's 100–200 block budgets fits, but run the
load sweep with `--blocks 1000` instead of 2000.

---

## 0. Prerequisites (done in NEW_BOX_SETUP.md)
- One NVIDIA GPU (24/48/80 GB), driver with CUDA ≥ 13.0, Ubuntu 22.04/24.04, ≥ 16 GB RAM, ≥ 50 GB disk.
- Repo cloned:
  `git clone https://github.com/MostafaaElhadidy/Agentic-KV-Cache-Management.git && cd Agentic-KV-Cache-Management`.
- Python 3.12 venv with `vllm==0.31.0` + `requirements-extra.txt`, activated in every shell.
- Llama-3.1-8B-Instruct downloaded: `hf download unsloth/Llama-3.1-8B-Instruct` (ungated mirror; the name is set
  once in `configs/models/llama31_8b.yaml`, switch it to `meta-llama/...` when access is approved).
- Work inside `tmux` so runs survive a disconnect.

## 1. Preflight (read-only) and choose the profile
```bash
bash scripts/box_preflight.sh                  # must end with "0 FAIL"; prints the suggested profile
HW=configs/hardware/box_48gb.yaml              # <- the suggested one
export HARDWARE=$HW                            # used by run_real_campaign.sh
```

## 2. Data, traces, tests, fingerprint check (no GPU, ~5 min)
```bash
bash scripts/fetch_gsm8k.sh
python scripts/make_traces.py --config configs/traces/cloud.yaml
pytest -q
python scripts/check_fingerprints.py           # must say PASS (Llama 3.1 adds a shared date header)
```
If `check_fingerprints.py` fails, raise `fingerprint_blocks` in `configs/experiments/real/cloud.yaml` as it says
(NEW_BOX_SETUP.md §8).

## 3. Sanity checks on the GPU (~10 min)
```bash
scripts/gpu_run.sh m1check 1800 python scripts/check_prefix_metrics.py \
  --config configs/experiments/m1_metrics_check/cloud.yaml --hardware $HW      # expect: OVERALL: PASS (10/10)
scripts/gpu_run.sh smoke_v 1800 python -m cachescout.run --config configs/experiments/smoke/cloud.yaml \
  --system vanilla --hardware $HW
scripts/gpu_run.sh smoke_c 1800 python -m cachescout.run --config configs/experiments/smoke/cloud.yaml \
  --system cachescout --hardware $HW
```
Read vLLM's "Available KV cache memory" and "GPU KV cache size" lines in `logs/smoke_v.log` to see the real
headroom of your card.

## 4. Verify the hook on this machine (sim-vs-GPU, ~30 min)
```bash
for s in vanilla lru_hook eviction_only; do
  scripts/gpu_run.sh xc_$s 2400 python -m cachescout.run --config configs/experiments/tune_gpu/cloud.yaml \
    --system $s --mode sequential --tag seq --hardware $HW
  python scripts/crosscheck_sim_vs_gpu.py results/tune_gpu/selector_tune/$s/b100_seq/result.json
done
```
Expected: `vanilla` and `lru_hook` → `exact_fraction: 1.0`; `eviction_only` → ≥ 0.95. If not, stop and compare
with the laptop's crosscheck files (`results/tune_gpu/*/b100_seq/crosscheck.json`).

## 5. Synthetic-trace experiments
```bash
CFG=configs/experiments/main/cloud.yaml
# (a) cache-size sweep (paper Fig. 14) + mechanism ablation (Fig. 12) + baselines (Fig. 8)
python scripts/compare.py --config $CFG --hardware $HW --blocks 100,150,200 \
  --systems vanilla,continuum,warmup_only,eviction_only,cachescout,cachescout_literal,no_prediction --run
# (b) seeds
for t in selector_eval_s2 selector_eval_s3; do
  python scripts/compare.py --config $CFG --hardware $HW --trace results/traces_cloud/$t.json \
    --blocks 100,150,200 --systems vanilla,eviction_only,cachescout --run; done
# (c) topologies (Fig. 4/5)
for t in pipeline_eval debate_eval random_eval; do
  python scripts/compare.py --config $CFG --hardware $HW --trace results/traces_cloud/$t.json \
    --blocks 100 --systems vanilla,cachescout --run; done
# (d) load sweep (Fig. 11): 1/5/20/50 sessions/s at a large budget (use --blocks 1000 on a 24 GB card)
for t in selector_eval selector_eval_r5 selector_eval_r20 selector_eval_r50; do
  python scripts/compare.py --config $CFG --hardware $HW --trace results/traces_cloud/$t.json \
    --blocks 2000 --systems vanilla,cachescout --run; done
```
Each run writes `results/main/<trace>/<system>/b<blocks>_eval/result.json`; each compare call writes a table
(`.md`), data (`.json`) and plot (`.png`) under `results/main/compare/`.

## 6. Real multi-agent experiments (GSM8K, Llama-3.1-8B agents)
Same stages as on the laptop; results go to `results/real_cloud/` (the laptop's are in `results/real/`).
Each stage is resumable and runs one GPU job at a time through `gpu_run.sh`.
```bash
export HARDWARE=$HW
bash scripts/run_real_campaign.sh tune_record cloud    # vanilla recordings on TRAIN problems (4 runs)
bash scripts/run_real_campaign.sh xcheck cloud         # sequential replay vs simulator (expect exact matches)
python scripts/tune_real.py --config configs/experiments/real/cloud.yaml   # optional re-tuning, see below
bash scripts/run_real_campaign.sh eval_record cloud    # 12 live vanilla recordings @100 blocks (test seeds 1-3)
bash scripts/run_real_campaign.sh replay cloud         # HEADLINE: 72 replay runs
bash scripts/run_real_campaign.sh live cloud           # 60 more live runs (LIVE_BLOCKS=100,150 to shorten)
python scripts/show_session.py results/real_cloud/gsm8k_test_selector_s1/vanilla/b100_eval   # read a session
```
- `tune_real.py` reports whether re-tuned constants beat the current ones by ≥ 0.2 pp on the Llama tuning
  recordings. It doesn't change any config; adopting new constants is a decision to log in `docs/decisions.md`.
- Aggregate the box campaign with `python scripts/aggregate_real.py --exp real_cloud`, which writes
  `results/report_real_cloud/` (the laptop's stays in `results/report_real/`).
- Expect Llama-3.1-8B to follow the tool/routing protocols better than the 1.5B model (fewer selector
  fallbacks). That changes the workload, so compare trends, not exact numbers.

## 7. Bring results home
On your laptop:
```bash
rsync -av <user>@<box-address>:Agentic-KV-Cache-Management/results/ ~/box_results/
rsync -av <user>@<box-address>:Agentic-KV-Cache-Management/logs/ ~/box_logs/
```
If the box is a paid cloud machine, **shut it down** afterwards (billing continues while it runs).

## 8. What to compare
- **Synthetic traces:** the same trends as `docs/REPORT.md` §4, now at the paper's model scale.
  - CacheScout vs vanilla hit rate per budget. Paper Fig. 14a: CacheScout flat at 86–87%, vanilla 64.4%→76.6%.
  - TTFT and latency reductions (Fig. 8b, 10a).
  - Eviction-only vs warmup-only (Fig. 12).
  - Random topology ≈ no gain.
  - Throughput vs load (Fig. 11).
- **Real agents:** replay vs live as in `docs/REPORT_REAL_AGENTS.md`.
  - With a costlier prefill (8B), does the hit-rate gain now turn into TTFT gains?
  - Does a stronger model reduce the selector fallback rate?
