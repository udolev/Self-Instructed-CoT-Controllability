# vLLM Patches

These patches target `vllm==0.17.1` exactly.

They are used for GPT-OSS reasoning prefill support and the reproduction vLLM environment:

```bash
./setup.sh --vllm
```

or manually:

```bash
bash vllm_patches/apply.sh .venv-vllm
```

The patches provide two things for this vLLM version:

- **GPT-OSS reasoning prefilling**: lets GPT-OSS be served with a custom chat template that enables reasoning prefilling.
- **A small Hermes tool-parser fix** for this vLLM version.

Do not apply these files to another vLLM version unless you have checked the
corresponding upstream files. If you use a newer vLLM version and your model supports custom chat templates, prefer serving
the model normally with a custom chat template that supports reasoning prefill.
