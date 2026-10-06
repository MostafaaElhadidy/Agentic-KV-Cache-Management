# Cloud runbook (M6): run the replication on a rented A100/H100

Everything here is for **you** to run on a cloud machine; nothing in this repo accesses any cloud account.
The code is the same as locally. Only `configs/hardware/cloud.yaml`, `configs/traces/cloud.yaml` and the
`configs/experiments/*/cloud.yaml` files differ (model, memory settings, workload size).

**What changes vs local:** model Llama-3.1-8B-Instruct (the paper's main model, Sec. 5.1) instead of
Qwen2.5-1.5B; `enforce_eager: false` (CUDA graphs on); `gpu_memory_utilization: 0.90`; `max_num_seqs: 256`;
traces with 300 sessions and arrival rates up to 50 sessions/s (paper Fig. 11 range).
Algorithm constants are **identical** to local (tuned on tuning traces, docs/decisions.md).

Rough budget: one A100-80GB or H100-80GB, ~4-8 GPU hours for everything below.

---

## 0. Pick a machine
- 1× A100 80 GB or H100 80 GB, Ubuntu 22.04/24.04, NVIDIA driver recent enough for CUDA 13
  (`nvidia-smi` top-right "CUDA Version" must be ≥ 13.0, because vLLM 0.31.0 wheels pull torch built for CUDA 13).
  If the driver is older, choose another image, or use the official container in step 2b.
- ≥ 100 GB disk, ≥ 64 GB RAM.

## 1. Get the code onto the machine
On your laptop (WSL), from the repo folder, make an archive with results excluded:
```bash
cd ~/CacheScoutImplementation
git archive --format=tar.gz -o /tmp/cachescout.tar.gz HEAD
scp /tmp/cachescout.tar.gz <user>@<cloud-ip>:~/
```
On the cloud machine:
```bash
mkdir -p ~/CacheScoutImplementation && tar -xzf ~/cachescout.tar.gz -C ~/CacheScoutImplementation
cd ~/CacheScoutImplementation
```

## 2a. Python environment (pinned to the local versions)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv ~/venv --python 3.12 && source ~/venv/bin/activate
uv pip install vllm==0.31.0            # same vLLM as local; brings its own torch build
uv pip install -r requirements-extra.txt
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```
Expected: `0.31.0 2.13.0+cu13x True`. **If vLLM is not 0.31.0, stop**: the hook was verified only on 0.31.0
(docs/vllm_internals.md has the file:line references to re-check).

## 2b. (Alternative) official container
```bash
docker run --gpus all -it --rm -v ~/CacheScoutImplementation:/work -w /work \
  --entrypoint bash vllm/vllm-openai:v0.31.0
pip install -r requirements-extra.txt
```

## 3. Model access (Llama is gated)
Request access to `meta-llama/Llama-3.1-8B-Instruct` on huggingface.co with **your** account, then:
```bash
hf auth login            # paste your own token (never commit it)
python -c "from huggingface_hub import snapshot_download as d; d('meta-llama/Llama-3.1-8B-Instruct')"
```

## 4. Sanity checks (≈10 min)
```bash
pytest -q                                                    # 85+ tests, no GPU needed
python scripts/make_traces.py --config configs/traces/cloud.yaml
python scripts/check_prefix_metrics.py --config configs/experiments/m1_metrics_check/cloud.yaml
# expect: OVERALL: PASS (10/10 checks)
python -m cachescout.run --config configs/experiments/smoke/cloud.yaml --system vanilla
python -m cachescout.run --config configs/experiments/smoke/cloud.yaml --system cachescout
```

## 5. Verify the hook on this machine (sim-vs-GPU, ≈30 min)
```bash
for s in vanilla lru_hook eviction_only; do
  python -m cachescout.run --config configs/experiments/tune_gpu/cloud.yaml --system $s --mode sequential --tag seq
  python scripts/crosscheck_sim_vs_gpu.py results/tune_gpu/selector_tune/$s/b100_seq/result.json
done
```
Expected: `vanilla` and `lru_hook` → `exact_fraction: 1.0`; `eviction_only` → ≥ 0.95. If not, stop and compare
with the local crosscheck files (results/tune_gpu/*/b100_seq/crosscheck.json).

## 6. Main experiments
```bash
CFG=configs/experiments/main/cloud.yaml
# (a) cache-size sweep like paper Fig. 14 + mechanism ablation (Fig. 12) + baselines (Fig. 8)
python scripts/compare.py --config $CFG --blocks 100,150,200 \
  --systems vanilla,continuum,warmup_only,eviction_only,cachescout,cachescout_literal,no_prediction --run
# (b) seeds
for t in selector_eval_s2 selector_eval_s3; do
  python scripts/compare.py --config $CFG --trace results/traces_cloud/$t.json --blocks 100,150,200 \
    --systems vanilla,eviction_only,cachescout --run; done
# (c) topologies (Fig. 4/5: gains should vanish under Random)
for t in pipeline_eval debate_eval random_eval; do
  python scripts/compare.py --config $CFG --trace results/traces_cloud/$t.json --blocks 100 \
    --systems vanilla,cachescout --run; done
# (d) load sweep (Fig. 11): arrival rate 1/5/20/50 sessions/s at a natural (non-overridden) budget
for t in selector_eval selector_eval_r5 selector_eval_r20 selector_eval_r50; do
  python scripts/compare.py --config $CFG --trace results/traces_cloud/$t.json --blocks 2000 \
    --systems vanilla,cachescout --run; done
```
Each run writes `results/main/<trace>/<system>/b<blocks>_eval/result.json`; each compare call writes a
table (`.md`), data (`.json`) and plot (`.png`) under `results/main/compare/`.

Optional, closer to the paper: a larger GPU budget per run (remove `num_gpu_blocks_override` from the experiment
config so vLLM sizes the cache from `gpu_memory_utilization`), and bigger anchors (`scale 1.0` profile in
`scripts/calibrate_workload.py`, then re-calibrate vanilla on the tuning trace for the new cache size).

## 7. Bring results home
On your laptop:
```bash
rsync -av <user>@<cloud-ip>:~/CacheScoutImplementation/results/ ~/CacheScoutImplementation/results_cloud/
rsync -av <user>@<cloud-ip>:~/CacheScoutImplementation/logs/ ~/CacheScoutImplementation/logs_cloud/
```
Then **shut the cloud machine down** (billing continues while it runs).

## 8. What to compare
Same trends as docs/REPORT.md §4, now at paper model scale: CacheScout vs vanilla hit rate per budget (paper
Fig. 14a: CacheScout flat 86-87%, vanilla 64.4%→76.6%), TTFT/latency reductions (Fig. 8b, 10a), eviction-only vs
warmup-only (Fig. 12), Random topology ≈ no gain, throughput vs load (Fig. 11).
