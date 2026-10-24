#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

show_help() {
  cat <<'EOF'
Usage: ./setup.sh [--agent] [--vllm]

Creates the project environments with uv.

Options:
  --agent   Install the Agent pipeline dependencies into .venv.
  --vllm    Create .venv-vllm with vLLM 0.17.1 and apply vllm_patches.
  -h, --help
            Show this help.

Examples:
  ./setup.sh
  ./setup.sh --agent
  ./setup.sh --vllm
  ./setup.sh --agent --vllm
EOF
}

install_agent=false
install_vllm=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      install_agent=true
      ;;
    --vllm)
      install_vllm=true
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
  shift
done

if ! command -v uv >/dev/null; then
  echo "error: uv is not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

main_sync_args=(sync --locked --python 3.12)
if [[ "$install_agent" == true ]]; then
  main_sync_args+=(--extra agent)
fi

echo "==> syncing .venv"
UV_PROJECT_ENVIRONMENT=.venv uv "${main_sync_args[@]}"

if [[ "$install_vllm" == true ]]; then
  echo "==> syncing .venv-vllm"
  UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --python 3.12 --extra vllm

  echo "==> applying vllm_patches to .venv-vllm"
  bash vllm_patches/apply.sh .venv-vllm
fi

echo
echo "done."
echo "  main env: source .venv/bin/activate"
if [[ "$install_vllm" == true ]]; then
  echo "  vLLM env: source .venv-vllm/bin/activate"
fi
