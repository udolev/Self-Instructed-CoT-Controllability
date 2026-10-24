#!/usr/bin/env bash
# Apply vllm_patches to a target venv's installed vllm package.
#
# Usage:
#   bash vllm_patches/apply.sh <path-to-venv>
#
# Example:
#   bash vllm_patches/apply.sh .venv-vllm
#
# The patches target vLLM 0.17.1 exactly. If you bump vLLM, verify the
# # PATCH START / # PATCH END markers in the patched files still map onto
# the same upstream functions before blindly overwriting. See
# vllm_patches/README.md for details.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path-to-venv>" >&2
  exit 1
fi

VENV="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ ! -d "$VENV" ]]; then
  echo "error: venv path not found: $VENV" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: no python executable at $PYTHON" >&2
  exit 1
fi

VLLM_DIR=$("$PYTHON" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null) || {
  echo "error: vllm not installed in $VENV (did you run './setup.sh --vllm'?)" >&2
  exit 1
}

VLLM_VERSION=$("$PYTHON" -c 'import vllm; print(vllm.__version__)')
if [[ "$VLLM_VERSION" != "0.17.1" ]]; then
  echo "warning: patches target vLLM 0.17.1; installed is $VLLM_VERSION" >&2
  echo "         verify PATCH START / END markers still map onto the same" >&2
  echo "         upstream functions before trusting this apply." >&2
fi

# Patched files — mirror the upstream tree.
FILES=(
  "vllm/entrypoints/openai/parser/harmony_utils.py"
  "vllm/entrypoints/openai/chat_completion/serving.py"
  "vllm/entrypoints/openai/engine/serving.py"
  "vllm/tool_parsers/hermes_tool_parser.py"
)

for rel in "${FILES[@]}"; do
  src="$SCRIPT_DIR/$rel"
  # strip leading "vllm/" since $VLLM_DIR already is .../site-packages/vllm
  dest="$VLLM_DIR/${rel#vllm/}"

  if [[ ! -f "$src" ]]; then
    echo "error: missing patched source: $src" >&2
    exit 1
  fi
  if [[ ! -f "$dest" ]]; then
    echo "error: missing upstream target: $dest" >&2
    exit 1
  fi

  cp "$src" "$dest"
  echo "patched: $dest"
done

echo "done. restart any running vLLM server to pick up the patches."
