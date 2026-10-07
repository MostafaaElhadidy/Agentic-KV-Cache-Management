"""Print whether the six agents have distinct plugin fingerprints for a model's chat template.

    python scripts/check_fingerprints.py                                   # Llama-3.1-8B-Instruct
    python scripts/check_fingerprints.py --model Qwen/Qwen2.5-7B-Instruct  # ungated fallback
    python scripts/check_fingerprints.py --config configs/experiments/real/cloud.yaml

Tokenizer only (no GPU, no weights); uses the local Hugging Face cache only (no download).
If it reports a collision, raise `cachescout.params.fingerprint_blocks` in the experiment config
to the printed `min_blocks_needed` (see docs/NEW_BOX_SETUP.md).
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.agents.fingerprints import fingerprint_report  # noqa: E402
from cachescout.agents.prompting import hf_chat_tokenizer  # noqa: E402
from cachescout.agents.routing import TOPOLOGIES  # noqa: E402
from cachescout.config import load_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/real/cloud.yaml")
    ap.add_argument("--model", default=None, help="override the config's model")
    args = ap.parse_args()
    exp = load_experiment(REPO / args.config, repo_root=REPO)
    model = args.model or exp["hardware_cfg"]["model"]["name"]
    blocks = int(exp["cachescout"]["params"].get("fingerprint_blocks", 2))
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer for {model} not in the local cache ({type(e).__name__}); "
              f"download it first: hf download {model}")
        return 2
    t = hf_chat_tokenizer(tok)
    ok = True
    print(f"model {model}, fingerprint_blocks={blocks} ({blocks * 16} tokens)")
    for topo in TOPOLOGIES:
        r = fingerprint_report(t, topo, blocks)
        ok &= r.distinct
        last = max(r.first_divergence.values())
        print(f"  {topo:<9} shared header/prefix {r.shared_prefix_tokens:>3} tokens, "
              f"last divergence at token {last:>3}, distinct={r.distinct}, "
              f"min_blocks_needed={r.min_blocks_needed}")
    ids = t([{"role": "system", "content": "X"}, {"role": "user", "content": "Task: x"}])
    print(f"  template start: {tok.decode(ids[:40])!r}")
    print("RESULT:", "PASS (six distinct fingerprints)" if ok else
          "FAIL: raise cachescout.params.fingerprint_blocks to min_blocks_needed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
