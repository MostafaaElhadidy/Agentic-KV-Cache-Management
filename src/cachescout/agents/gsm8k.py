"""GSM8K tasks (github.com/openai/grade-school-math, MIT): loading, disjoint selection, scoring."""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "gsm8k"
MASTER_SEED = 20261006   # fixed permutation; seed k takes slice k, so seeds never share problems


@dataclass(frozen=True)
class Problem:
    split: str
    index: int          # line index in the official jsonl (0-based) = problem id
    question: str
    gold: str           # number after '####'

    @property
    def problem_id(self) -> str:
        return f"{self.split}:{self.index}"


def gold_answer(answer_field: str) -> str:
    """GSM8K gold answer: the text after '####', commas removed."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[Problem]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: bash scripts/fetch_gsm8k.sh")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            out.append(Problem(split, i, row["question"], gold_answer(row["answer"])))
    return out


def select_problems(problems: list[Problem], seed: int, n: int) -> list[Problem]:
    """Disjoint by construction: one fixed permutation of the split, seed k (>= 1) takes the k-th
    slice of length n. Tuning uses the train split, evaluation the test split (engineering
    choice)."""
    if seed < 1:
        raise ValueError("problem seeds start at 1")
    order = list(range(len(problems)))
    random.Random(MASTER_SEED).shuffle(order)
    start = (seed - 1) * n
    if start + n > len(order):
        raise ValueError("not enough problems for this seed/size")
    return [problems[i] for i in order[start:start + n]]


_NUM = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> str | None:
    """First number in `text` (commas and $ removed), or None."""
    m = _NUM.search(text)
    if not m:
        return None
    return m.group(0).replace("$", "").replace(",", "").replace(" ", "")


def is_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False
