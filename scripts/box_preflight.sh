#!/usr/bin/env bash
# Read-only preflight check for a remote box. Installs nothing, downloads nothing, changes nothing.
#
#   bash scripts/box_preflight.sh                 # uses `python` from the active environment
#   PYTHON=~/cachescout-venv/bin/python MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/box_preflight.sh
#
# Prints GPU / driver / CUDA, RAM, disk, Python + package versions, whether the model and tokenizer
# are already in the Hugging Face cache, GSM8K presence, and PASS / WARN / FAIL lines with reasons.
# It ends with a suggested hardware profile (configs/hardware/box_{24,48,80}gb.yaml).
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
HF_CACHE=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
fails=0; warns=0
pass() { echo "PASS  $*"; }
warn() { echo "WARN  $*"; warns=$((warns + 1)); }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # ver_ge A B: A >= B

echo "=== GPU ==="
gpu_mib=0
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
  # classic header "CUDA Version: 13.0" or newer drivers' "CUDA UMD Version: 13.3"
  cuda=$(nvidia-smi | grep -oE "CUDA (UMD )?Version: [0-9.]+" | head -1 | grep -oE "[0-9.]+$")
  echo "CUDA version supported by driver: ${cuda:-unknown}"
  gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  ngpu=$(nvidia-smi -L | wc -l)
  [ "$ngpu" -ge 1 ] && pass "found $ngpu GPU(s); GPU 0 has ${gpu_mib} MiB" || fail "no GPU listed by nvidia-smi"
  if [ -n "${cuda:-}" ] && ver_ge "$cuda" "13.0"; then pass "driver supports CUDA $cuda (>= 13.0 needed by vllm 0.31.0 wheels / torch cu13x)"
  else fail "driver CUDA ${cuda:-unknown} < 13.0: vllm 0.31.0 + torch 2.13 (cu13x) may not run; ask the admin for a newer driver"; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$used" -lt 1000 ] && pass "GPU 0 is idle (${used} MiB used)" || warn "GPU 0 already uses ${used} MiB (someone else running?)"
else
  fail "nvidia-smi not found: no NVIDIA driver visible on this box"
fi

echo; echo "=== Memory and disk ==="
free -h | head -2
ram_gib=$(free -g | awk '/^Mem:/{print $7}')
[ "$ram_gib" -ge 16 ] && pass "available RAM ${ram_gib} GiB (>= 16)" || warn "available RAM ${ram_gib} GiB (< 16 GiB recommended)"
df -h . "$HOME" 2>/dev/null | awk 'NR==1 || !seen[$0]++'
disk_gib=$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$disk_gib" -ge 50 ] && pass "free disk here ${disk_gib} GiB (>= 50)" || warn "free disk here ${disk_gib} GiB (< 50 GiB recommended; model ~16 GB + env ~10 GB)"
if [ -d "$HF_CACHE" ]; then
  hf_free=$(df -BG "$HF_CACHE" | awk 'NR==2{gsub("G","",$4); print $4}')
  echo "Hugging Face cache: $HF_CACHE (free ${hf_free} GiB)"
else
  echo "Hugging Face cache: $HF_CACHE (does not exist yet)"
fi

echo; echo "=== Python environment ($PY) ==="
if command -v "$PY" >/dev/null 2>&1; then
  pyv=$("$PY" -c 'import sys; print(sys.version.split()[0])')
  echo "python $pyv at $(command -v "$PY")"
  [[ "$pyv" == 3.12.* ]] && pass "Python $pyv" || warn "Python $pyv (project tested on 3.12)"
  pkgs=$("$PY" - <<'EOF'
from importlib.metadata import version, PackageNotFoundError
for p in ("vllm", "torch", "transformers", "pytest", "ruff", "pyyaml", "matplotlib", "huggingface_hub"):
    try:
        print(f"{p} {version(p)}")
    except PackageNotFoundError:
        print(f"{p} MISSING")
EOF
)
  echo "$pkgs"
  v=$(echo "$pkgs" | awk '$1=="vllm"{print $2}'); t=$(echo "$pkgs" | awk '$1=="torch"{print $2}')
  [ "$v" = "0.31.0" ] && pass "vllm 0.31.0" || fail "vllm is '$v' (need exactly 0.31.0; the plugin was verified only on it)"
  [[ "$t" == 2.13.* ]] && pass "torch $t" || warn "torch '$t' (local runs used 2.13.0+cu132)"
  echo "$pkgs" | grep -q "MISSING" && warn "some packages missing (pip install -r requirements-extra.txt)" || pass "all project packages present"
else
  fail "Python '$PY' not found (activate the venv or set PYTHON=...)"
fi

echo; echo "=== Model and tokenizer in the Hugging Face cache (no download) ==="
mdir="$HF_CACHE/models--${MODEL//\//--}"
snap=$(ls -d "$mdir"/snapshots/* 2>/dev/null | head -1)
if [ -n "$snap" ] && [ -f "$snap/config.json" ]; then
  pass "config.json cached for $MODEL"
  [ -f "$snap/tokenizer.json" ] || [ -f "$snap/tokenizer.model" ] && pass "tokenizer cached for $MODEL" || fail "tokenizer files missing for $MODEL"
  nweights=$(ls "$snap"/*.safetensors 2>/dev/null | wc -l)
  [ "$nweights" -ge 1 ] && pass "$nweights safetensors file(s) cached ($(du -shL "$snap" 2>/dev/null | cut -f1))" || fail "weights not cached for $MODEL (run: hf download $MODEL)"
  if command -v "$PY" >/dev/null 2>&1; then
    "$PY" - "$snap/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
L, kv = c["num_hidden_layers"], c.get("num_key_value_heads", c["num_attention_heads"])
hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
per_tok = 2 * L * kv * hd * 2
print(f"KV cache (bf16): {L} layers x {kv} KV heads x {hd} dim -> {per_tok} B/token = "
      f"{per_tok / 1024:.0f} KiB/token = {16 * per_tok / 2**20:.3f} MiB per 16-token block")
EOF
  fi
else
  fail "$MODEL not in the cache ($mdir). Llama is gated: request access, 'hf auth login', then 'hf download $MODEL'"
fi
[ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ] && pass "a Hugging Face token file exists (not printed)" || warn "no Hugging Face token file (needed for gated meta-llama models)"

echo; echo "=== Data ==="
if [ -f data/gsm8k/test.jsonl ] && [ -f data/gsm8k/train.jsonl ]; then
  (cd data/gsm8k && sha256sum -c --quiet SHA256SUMS) && pass "GSM8K present, checksums OK" || fail "GSM8K checksums do not match"
else
  warn "GSM8K not downloaded yet (bash scripts/fetch_gsm8k.sh)"
fi
[ -d results/traces_cloud ] && pass "cloud traces generated" || warn "cloud traces not generated yet (python scripts/make_traces.py --config configs/traces/cloud.yaml)"

echo; echo "=== Suggested hardware profile ==="
if [ "$gpu_mib" -ge 70000 ]; then echo "configs/hardware/box_80gb.yaml"
elif [ "$gpu_mib" -ge 40000 ]; then echo "configs/hardware/box_48gb.yaml"
elif [ "$gpu_mib" -ge 20000 ]; then echo "configs/hardware/box_24gb.yaml"
elif [ "$gpu_mib" -gt 0 ]; then echo "GPU < 20 GB: Llama-3.1-8B in bf16 does not fit; use configs/hardware/local.yaml (Qwen2.5-1.5B)"
else echo "(no GPU detected)"; fi

echo; echo "Summary: $fails FAIL, $warns WARN"
[ "$fails" -eq 0 ]
