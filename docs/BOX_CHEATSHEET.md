# Box cheat sheet: the 10 commands, in order

Details and troubleshooting: `docs/NEW_BOX_SETUP.md`. Experiments: `docs/CLOUD_RUNBOOK.md`.

```bash
# 1. connect (from your laptop), then start a tmux session that survives disconnects
ssh <user>@<box-address>
tmux new -s cs                       # detach: Ctrl+b then d    re-attach: tmux attach -t cs

# 2. get the code
git clone https://github.com/MostafaaElhadidy/Agentic-KV-Cache-Management.git && cd Agentic-KV-Cache-Management

# 3. Python 3.12 environment (no root)
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env && uv venv ~/cachescout-venv --python 3.12 && source ~/cachescout-venv/bin/activate

# 4. vLLM 0.31.0 + project extras
uv pip install vllm==0.31.0 && uv pip install -r requirements-extra.txt

# 5. model: ungated mirror of Llama-3.1-8B-Instruct, no login (name in configs/models/llama31_8b.yaml)
hf download unsloth/Llama-3.1-8B-Instruct

# 6. read-only preflight: must end with "0 FAIL"; note the suggested profile
bash scripts/box_preflight.sh

# 7. data + traces + tests + fingerprint check (no GPU)
bash scripts/fetch_gsm8k.sh && python scripts/make_traces.py --config configs/traces/cloud.yaml && pytest -q && python scripts/check_fingerprints.py

# 8. first GPU check (replace box_48gb with your profile)
export HARDWARE=configs/hardware/box_48gb.yaml
scripts/gpu_run.sh m1check 1800 python scripts/check_prefix_metrics.py --config configs/experiments/m1_metrics_check/cloud.yaml --hardware $HARDWARE

# 9. the real-agent campaign (one stage at a time; each is resumable)
for st in tune_record xcheck eval_record replay live; do bash scripts/run_real_campaign.sh $st cloud || break; done

# 10. summarise, then copy results to your laptop (run the rsync on the laptop)
python scripts/aggregate_real.py --exp real_cloud
rsync -av <user>@<box-address>:Agentic-KV-Cache-Management/results/ ~/box_results/
```

Every new shell: `cd Agentic-KV-Cache-Management && source ~/cachescout-venv/bin/activate`.
Never run two GPU jobs at once; `scripts/gpu_run.sh` refuses to start if one is already running.
