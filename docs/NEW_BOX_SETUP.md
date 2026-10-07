# Setting up a new Linux GPU box (no root needed)

This guide is for running the project on a remote Linux machine with one NVIDIA GPU of 24, 48 or 80 GB, using
**Llama-3.1-8B-Instruct** (the paper's main model). No root access is needed. Every step lists the command,
what you should see, and what to do if it fails. The one-page version is `docs/BOX_CHEATSHEET.md`.

Throughout, `$` lines are commands you type on the **box** (after `ssh`), unless they say "on your laptop".

---

## 1. Connect

```bash
# on your laptop (WSL terminal)
ssh <user>@<box-address>
```
- **You should see** a shell prompt on the box.
- **If it fails:**
  - "Permission denied (publickey)": ask your instructor to add your SSH public key (`cat ~/.ssh/id_ed25519.pub`
    on your laptop; create one with `ssh-keygen -t ed25519` if missing).
  - "Connection timed out": check the address/VPN.

Then check the GPU:
```bash
nvidia-smi
```
- **You should see** a table with the GPU name (e.g. `NVIDIA L40S`, `A100-SXM4-80GB`), its memory, and the
  driver's CUDA version (top right: `CUDA Version: 13.x`, or `CUDA UMD Version` on newer drivers).
- **If `CUDA Version` is below 13.0:** vLLM 0.31.0 installs torch built for CUDA 13 and may not run. Ask the
  admin for a newer NVIDIA driver; you cannot fix this without root.

## 2. Keep runs alive when you disconnect (tmux)

```bash
tmux new -s cs            # start a session called "cs"
# ... run things ...
# detach: press Ctrl+b, release, then d        (runs keep going)
tmux attach -t cs         # come back later, even after a new ssh login
```
- **If `tmux` is missing:** try `screen -S cs` (detach: Ctrl+a then d; reattach: `screen -r cs`). If neither
  exists, run long jobs with `nohup <command> > logs/<name>.log 2>&1 &` and watch them with
  `tail -f logs/<name>.log`.

## 3. Get the code

```bash
git clone https://github.com/MostafaaElhadidy/Agentic-KV-Cache-Management.git
cd Agentic-KV-Cache-Management
git log --oneline -3
```
- **You should see** the latest commits.
- **If the files from this guide are missing** (for example `scripts/box_preflight.sh`), the `box-prep` branch
  isn't merged yet: run `git checkout box-prep`.

## 4. Install Python 3.12 with uv (no root)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin
source $HOME/.local/bin/env                         # or open a new shell
uv python install 3.12
uv venv ~/cachescout-venv --python 3.12
source ~/cachescout-venv/bin/activate               # do this in every new shell / tmux window
python --version
```
- **You should see** `Python 3.12.x`.
- **If `curl` is blocked:** ask whether the box allows outbound internet. Without it, you need the admin to
  provide Python 3.12 and a pip mirror.
- **If your home directory is small** (`df -h ~` shows < 50 GB free): create the venv on a bigger disk, e.g.
  `uv venv /scratch/$USER/cachescout-venv --python 3.12`.

## 5. Install vLLM 0.31.0 and the project's extras

```bash
uv pip install vllm==0.31.0             # brings its own torch (2.13, CUDA 13 build); several GB
uv pip install -r requirements-extra.txt
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.is_available())"
```
- **You should see** `0.31.0 2.13.0+cu13x 5.x True`.
- **If `torch.cuda.is_available()` is `False`:** the driver is too old for this torch build (see step 1), or no
  GPU is visible (`echo $CUDA_VISIBLE_DEVICES`).
- **If vLLM isn't 0.31.0:** don't continue. The plugin was verified only on 0.31.0
  (`docs/vllm_internals.md` has the file:line references to re-check).

## 6. Hugging Face access and model download

`meta-llama/Llama-3.1-8B-Instruct` is **gated**: request access once, from a browser, with your own account.

1. Open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, log in, accept the license, and request
   access. Approval can take minutes to days.
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. On the box:
```bash
export HF_HOME=$HOME/.cache/huggingface          # or a bigger disk, e.g. /scratch/$USER/hf (use it everywhere)
hf auth login                                     # paste the token; never put it in a file in the repo
hf download meta-llama/Llama-3.1-8B-Instruct      # ~16 GB
```
- **You should see** files being downloaded, ending with a path under `.../models--meta-llama--Llama-3.1-8B-Instruct/snapshots/...`.
- **If you get "401/403 ... gated repo":** access isn't approved yet. Meanwhile use the **ungated fallback**:
```bash
hf download Qwen/Qwen2.5-7B-Instruct              # ~15 GB, Apache-2.0, no approval needed
```
  To use it, change one line in your GPU profile (§9), e.g. in `configs/hardware/box_48gb.yaml`:
  `name: Qwen/Qwen2.5-7B-Instruct`. Report any results as "Qwen2.5-7B", not as the paper's model.
- **If `hf` is not found** (it comes with vLLM's `huggingface_hub` dependency), use the Python API instead:
  `python -c "from huggingface_hub import login; login()"` and
  `python -c "from huggingface_hub import snapshot_download as d; d('meta-llama/Llama-3.1-8B-Instruct')"`.
- The box profiles set `HF_HUB_OFFLINE=1`: runs never download anything. If a run says the model isn't found,
  the download in this step didn't finish.

## 7. Preflight check (read-only)

```bash
bash scripts/box_preflight.sh
# fallback model: MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/box_preflight.sh
```
- **You should see** PASS lines for the GPU, the CUDA 13 driver, Python 3.12, vllm 0.31.0 and the cached
  model/tokenizer; the KV bytes per token read from the model's `config.json`; and a **suggested hardware
  profile** at the end.
- **Fix every FAIL before running GPU jobs.** Each line says why it failed. WARN lines are advice.

## 8. Data, traces and tests (no GPU)

```bash
bash scripts/fetch_gsm8k.sh                                   # GSM8K + checksum verification
python scripts/make_traces.py --config configs/traces/cloud.yaml
pytest -q
python scripts/check_fingerprints.py                          # Llama chat template vs plugin fingerprint
```
- **You should see:**
  - `sha256sum ... OK` for both GSM8K files;
  - `Saved results/traces_cloud/`;
  - all tests passing (tokenizer tests skip only if a tokenizer isn't downloaded);
  - `RESULT: PASS (six distinct fingerprints)`.
- **If `check_fingerprints.py` says FAIL:**
  - Llama 3.1's template puts a shared header ("Cutting Knowledge Date… Today Date…") before every system
    prompt. If that header is longer than the plugin's 32-token fingerprint window, the six agents would look
    like one.
  - Fix: set `fingerprint_blocks:` to the printed `min_blocks_needed` in the `cachescout: params:` of
    `configs/experiments/real/cloud.yaml`, then re-run the check.
  - Synthetic traces are not affected: they send raw token IDs without a chat template.

## 9. Pick the GPU profile

Use the profile `box_preflight.sh` suggests: `configs/hardware/box_24gb.yaml`, `box_48gb.yaml` or `box_80gb.yaml`.
Pass it to every run with `--hardware`, or with `HARDWARE=...` for the campaign script.

### Memory arithmetic for Llama-3.1-8B-Instruct (bf16)
From the model card / `config.json`: 32 layers, 8 KV heads (grouped-query attention), head dim 128,
8.03 B parameters. Computed by `cachescout.kv_blocks` (tested):

| Quantity | Formula | Value |
|---|---|---|
| Weights | 8.03 B × 2 bytes | **16.1 GB (14.96 GiB)** |
| KV bytes per token | 2 (K and V) × 32 layers × 8 KV heads × 128 dim × 2 bytes | **131,072 B = 128 KiB** |
| KV bytes per 16-token block | 16 × 128 KiB | **2 MiB** |
| Paper budgets 100 / 150 / 200 blocks | × 2 MiB | 0.20 / 0.29 / 0.39 GiB |
| One 1,584-token request (99 blocks) | | 0.19 GiB |

(Ungated fallback Qwen2.5-7B: 28 layers, 4 KV heads, dim 128 → 56 KiB per token, 0.875 MiB per block,
15.2 GB weights.)

### Estimated free KV memory per profile (estimates; vLLM's log line "Available KV cache memory" is the truth)
| Profile | Usable GPU (×0.90) | − weights | − activations/graphs (estimate) | ≈ KV memory | ≈ natural blocks |
|---|---|---|---|---|---|
| `box_24gb` (L4, A10G, RTX 3090/4090) | ~20.2 GiB | 15.0 | ~2.5 (eager) | **~2.7 GiB** | **~1,380** |
| `box_48gb` (L40S, RTX A6000) | ~40 GiB | 15.0 | ~3 | **~22 GiB** | **~11,000** |
| `box_80gb` (A100/H100 80 GB) | ~71 GiB | 15.0 | ~4 | **~52 GiB** | **~26,000** |

### Which experiments in `docs/CLOUD_RUNBOOK.md` fit
| Experiment | KV it needs | 24 GB | 48 GB | 80 GB |
|---|---|---|---|---|
| M1 prefix-cache check (256 blocks) | 0.5 GiB | ✅ | ✅ | ✅ |
| Synthetic cache-size sweep, seeds, topologies (100–200 blocks) | ≤ 0.4 GiB | ✅ | ✅ | ✅ |
| Synthetic load sweep at `--blocks 2000` | 3.9 GiB | ❌ (use `--blocks 1000`) | ✅ | ✅ |
| Real-agent campaign (replay + live, 100–200 blocks) | ≤ 0.4 GiB | ✅ (slower: eager mode) | ✅ | ✅ |
| Natural cache size (no override), many concurrent sessions | the whole KV pool | small (~1.4k blocks) | ✅ | ✅ (largest) |
| Paper's Qwen3-235B study | 4× H200 | ❌ | ❌ | ❌ |

On 24 GB, `enforce_eager: true` avoids CUDA-graph memory (as on the 8 GB laptop). If `nvidia-smi` shows ample
headroom after the first smoke run, you may set it to `false` for faster decoding. Apply the same setting to
**every** system you compare.

## 10. First GPU runs (short, in tmux)

```bash
HW=configs/hardware/box_48gb.yaml          # your profile from step 7
scripts/gpu_run.sh m1check 1800 python scripts/check_prefix_metrics.py \
    --config configs/experiments/m1_metrics_check/cloud.yaml --hardware $HW
scripts/gpu_run.sh smoke 1800 python -m cachescout.run --config configs/experiments/smoke/cloud.yaml \
    --system cachescout --hardware $HW
```
- **You should see:**
  - `OVERALL: PASS (10/10 checks)` in `logs/m1check.log`;
  - `[run] done ... hit_rate=...` in `logs/smoke.log`;
  - `logs/*.monitor.log` shows GPU memory below the card's size.
- **If out of memory:** lower `max_num_seqs` or `gpu_memory_utilization` in the profile, or set
  `enforce_eager: true`.

Then follow `docs/CLOUD_RUNBOOK.md` for the experiments.

## 11. Copy results back to your laptop

```bash
# on your laptop
rsync -av <user>@<box-address>:Agentic-KV-Cache-Management/results/ ~/box_results/
rsync -av <user>@<box-address>:Agentic-KV-Cache-Management/logs/ ~/box_logs/
```
