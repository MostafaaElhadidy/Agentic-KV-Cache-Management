#!/usr/bin/env bash
# PostToolUse hook: lint only the Python file Claude just edited. Fast; never touches the GPU.
# Exit 2 sends ruff's findings back to Claude so it can fix them.
file=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
[[ "$file" == *.py && -f "$file" ]] || exit 0
RUFF="$HOME/testLLM/.venv/bin/ruff"
[[ -x "$RUFF" ]] || exit 0
if ! out=$("$RUFF" check --quiet "$file" 2>&1); then
  echo "ruff found issues in $file:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
