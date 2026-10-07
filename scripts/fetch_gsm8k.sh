#!/usr/bin/env bash
# Download GSM8K (MIT license, https://github.com/openai/grade-school-math) into data/gsm8k/ and verify
# SHA-256 checksums. data/ is git-ignored; the checksums are tracked in data/gsm8k/SHA256SUMS.
#   bash scripts/fetch_gsm8k.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=data/gsm8k
BASE=https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data
mkdir -p "$DIR"
for f in test.jsonl train.jsonl; do
  if [ ! -s "$DIR/$f" ]; then
    echo "downloading $f"
    curl -fsSL --retry 3 -o "$DIR/$f.part" "$BASE/$f"
    mv "$DIR/$f.part" "$DIR/$f"
  fi
done
if [ -f "$DIR/SHA256SUMS" ]; then
  (cd "$DIR" && sha256sum -c SHA256SUMS)
else
  (cd "$DIR" && sha256sum test.jsonl train.jsonl > SHA256SUMS)
  echo "recorded new checksums in $DIR/SHA256SUMS"
fi
wc -l "$DIR"/test.jsonl "$DIR"/train.jsonl
