# Backend Support and Local vLLM Serving

OpenRouter runs need no local server. Use local vLLM to serve a model yourself, or to run reasoning-prefill methods with models that OpenRouter cannot prefill (GPT-OSS in the paper).

Commands below are run from the repository root.

## Table of Contents

| Section | Description |
|---------|-------------|
| [Method Support](#method-support) | Which elicitation methods need reasoning-prefill support |
| [Setup](#setup) | Client and vLLM environments |
| [Serve a Model](#serve-a-model) | vLLM serving examples |
| [Run Against the Server](#run-against-the-server) | QA and Agent client commands |
| [Chat Templates](#chat-templates) | Reasoning-prefill template files |
| [Newer vLLM](#newer-vllm) | Using a different vLLM version |

---

## Method Support

| Method flag | Name | Works on |
|-------------|------------|----------|
| `user_prompt` | User Prompt | any backend that returns the reasoning trace |
| `constraint_bootstrap_prompt` | Constraint Bootstrap | any backend that returns the reasoning trace; the model needs a prefill family |
| `generic_bootstrap_prompt` | Generic Bootstrap | any backend that returns the reasoning trace; the model needs a prefill family |
| `constraint_prefill` | Constraint Prefill | a backend that can prefill the reasoning channel; the model needs a prefill family |
| `user_prompt_constraint_prefill` | User Prompt + Constraint Prefill | a backend that can prefill the reasoning channel; the model needs a prefill family |
| `user_prompt_generic_prefill` | User Prompt + Generic Prefill | a backend that can prefill the reasoning channel; the model needs a prefill family |

Reasoning-prefill support in this repo:

| Model family | OpenRouter | Local vLLM |
|--------------|:----------:|:----------:|
| Qwen3 | yes | yes |
| DeepSeek-R1 | yes | yes |
| GPT-OSS | no | yes |
| Claude | no | no; use `user_prompt` or the bootstrap methods |

For a new model, see [docs/adding_a_model.md](adding_a_model.md).

---

## Setup

| Need | Command |
|------|---------|
| QA client only | `./setup.sh` |
| Agent client only | `./setup.sh --agent` |
| Local vLLM server | `./setup.sh --vllm` |
| Agent client + local vLLM server | `./setup.sh --agent --vllm` |

`setup.sh` always syncs `.venv` first. With `--vllm`, it also creates `.venv-vllm`, installs `vllm==0.17.1`, and applies [vllm_patches/](../vllm_patches/). The vLLM environment is separate from `.venv` because the pinned serving stack and the Agent client dependencies have incompatible requirements.

---

## Serve a Model

### GPT-OSS

QA serving:

```bash
.venv-vllm/bin/vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --chat-template chat_templates/gpt_oss_open_reasoning_prefill.jinja
```

Agent serving adds tool-calling support:

```bash
.venv-vllm/bin/vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --chat-template chat_templates/gpt_oss_open_reasoning_prefill.jinja
```

The client's default `--reasoning-effort medium` is the paper's GPT-OSS setting. Larger models may need tensor-parallel, expert-parallel, KV-cache, or model-length flags appropriate to your hardware.

### Qwen3 and DeepSeek-R1

Use the model-family template and parser flags supported by vLLM:

| Family | Template | Common parser flags |
|--------|----------|---------------------|
| Qwen3 | `chat_templates/qwen3_open_reasoning_prefill.jinja` | `--reasoning-parser qwen3`; Agent usually also uses `--enable-auto-tool-choice --tool-call-parser hermes` |
| DeepSeek-R1 | `chat_templates/deepseek_r1_open_reasoning_prefill.jinja` | `--reasoning-parser deepseek_r1` |
| DeepSeek-R1 0528 Agent | `chat_templates/deepseek_r1_0528_open_reasoning_prefill.jinja` | `--reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser deepseek_v3` |

---

## Run Against the Server

### QA

```bash
uv run python SICC-QA/run_cceval.py \
  --model openai/gpt-oss-20b \
  --api-base http://127.0.0.1:8000/v1 \
  --mode uppercase_thinking \
  --elicitation-method constraint_prefill
```

### Agent

Prefix the model with `vllm/` so inspect_ai routes requests to the local vLLM backend:

```bash
uv run --extra agent python SICC-Agent/run_bfcl.py \
  --model vllm/openai/gpt-oss-20b \
  --api-base http://127.0.0.1:8000/v1 \
  --mode uppercase_thinking \
  --elicitation-method constraint_prefill \
  --limit 10
```

---

## Chat Templates

On local vLLM, reasoning-prefill methods send a final assistant message whose visible `content` is empty and whose `reasoning` field contains the prefill. The chat template must open the model's reasoning block, write that prefill, and leave the reasoning block open so generation continues inside it.

| Model family | Template |
|--------------|----------|
| GPT-OSS | `chat_templates/gpt_oss_open_reasoning_prefill.jinja` |
| Qwen3 | `chat_templates/qwen3_open_reasoning_prefill.jinja` |
| DeepSeek-R1 | `chat_templates/deepseek_r1_open_reasoning_prefill.jinja` |
| DeepSeek-R1 0528 | `chat_templates/deepseek_r1_0528_open_reasoning_prefill.jinja` |

Prompt and bootstrap methods do not require a special chat template.

---

## Newer vLLM

You can use a newer vLLM instead of the pinned environment. Do not apply `vllm_patches/` to another vLLM version unless you have checked the corresponding upstream files. You still need a chat template that supports the reasoning-prefill format above.
