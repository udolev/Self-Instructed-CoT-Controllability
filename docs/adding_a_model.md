# Adding a Model

What a new model needs depends on the elicitation methods you want to run. Commands use repository-root paths.

## Table of Contents

| Section | Description |
|---------|-------------|
| [Requirements by Method](#requirements-by-method) | What each method needs |
| [Serve the Model](#serve-the-model) | Backend and model id (required) |
| [Choose a Prefill Family](#choose-a-prefill-family) | Required for bootstrap and reasoning-prefill methods |
| [Add a Chat Template](#add-a-chat-template) | Required for reasoning prefill on local vLLM with a new reasoning format |
| [Check the Model](#check-the-model) | Smoke test and what to inspect |

---

## Requirements by Method

| Method | Needs |
|--------|-------|
| `user_prompt` | A backend that returns the reasoning trace |
| `constraint_bootstrap_prompt`, `generic_bootstrap_prompt` | The above, plus a prefill family for the model |
| `constraint_prefill`, `user_prompt_constraint_prefill`, `user_prompt_generic_prefill` | The above, plus a backend that can prefill the reasoning channel |

Reasoning prefill through OpenRouter is implemented only for Qwen3 and DeepSeek-R1. Any other model must be served locally with vLLM, using a chat template that opens the reasoning block (see [Add a Chat Template](#add-a-chat-template)). The current backend matrix is in [serving.md](serving.md#method-support).

---

## Serve the Model

Required for every method.

**OpenRouter.** Pass the OpenRouter model id: `--model <provider/model>` for QA and `--model openrouter/<provider/model>` for Agent. The reasoning-token budget is requested only when the model id contains one of the substrings in `REASONING_MODEL_KEYWORDS` (`prefill.py`); if the new model's id does not, add a substring to that tuple in both `SICC-QA/prefill.py` and `SICC-Agent/prefill.py`.

**Local vLLM.** Serve the model with vLLM and pass `--api-base http://127.0.0.1:8000/v1` (QA: `--model <model>`; Agent: `--model vllm/<model>`). For Agent tasks, also enable the model's vLLM tool-calling parser. Serving commands are in [serving.md](serving.md).

With the model served, `user_prompt` runs without code changes:

```bash
uv run python SICC-QA/run_cceval.py --model <provider/model> --test
```

---

## Choose a Prefill Family

Required for the bootstrap and reasoning-prefill methods, which render the self-instruction from the templates in `prefill.py`. A model is mapped to a family by `MODEL_FAMILY_PATTERNS`:

| Family | Used for | Style |
|--------|----------|-------|
| `deepseek_qwen` | DeepSeek-R1, Qwen3, Claude | Conversational ("Okay, let's tackle this problem. Hmm, I need to ...") |
| `gpt_oss` | GPT-OSS | Short and declarative ("Solving while ...") |

**Reuse an existing family** (usually enough): add a substring of the model id to the family's entry in `MODEL_FAMILY_PATTERNS`, in both `SICC-QA/prefill.py` and `SICC-Agent/prefill.py`. The model then uses that family's self-instructions.

**Add a new family** (optional, when neither style fits the model's reasoning): in both `prefill.py` files, add the family to `MODEL_FAMILY_PATTERNS` (and to the `PREFILL_MODEL_FAMILIES` list), a constraint-specific template for each constraint to `PREFILL_PER_CONSTRAINT_TEMPLATES`, and a generic template to `FIXED_PREFILL_TEMPLATES`. Keep the self-instruction short and written as the model beginning its own reasoning, not as an external instruction. For the constraints on the surface form of the CoT, the template is styled automatically (uppercased, "meow" inserted, and so on).

---

## Add a Chat Template

Required for reasoning-prefill methods on local vLLM when the model's reasoning format is not covered by the shipped templates. On local vLLM, reasoning-prefill methods send the self-instruction as a final assistant message with an empty `content` and the text in a `reasoning` field:

```python
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "reasoning": "Solving while following the constraint. "},
]
```

The chat template must detect this message, open the model's reasoning block, write the text, and leave the block open so generation continues inside it. A new template is usually the model's own chat template plus this one branch; the templates under `chat_templates/` (Qwen3, DeepSeek-R1, GPT-OSS) show the pattern. Pass the template with `--chat-template` when serving (see [serving.md](serving.md#chat-templates)).

---

## Check the Model

Run a 3-sample smoke test with the method you want, then grade it:

```bash
uv run python SICC-QA/run_cceval.py \
  --model <provider/model> \
  --mode uppercase_thinking \
  --elicitation-method <method> \
  --test

uv run python SICC-QA/grade_compliance_csv.py \
  --input_file logs/results/<run>.csv \
  --include-soft
```

Inspect a few rows of the graded CSV:

| Column | What to check |
|--------|---------------|
| `reasoning_prefill` | The exact self-instruction you intended (reasoning-prefill methods) |
| `reasoning` | Begins with the model's own continuation, or with the prefill or echoed self-instruction if the backend returned it (the grader strips it either way) |
| `control_value` | The target the grader should use |
| `compliance` | Agrees with your own judgment on a few examples |
