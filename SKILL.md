---
name: vibe-abliteration
description: "Automate model censorship removal (abliteration) with Heretic - install dependencies, run optimization, evaluate trade-offs (refusals vs KL divergence), save/upload decensored models with proper model cards. Use when asked to 'abliterate', 'decensor', 'uncensor', or 'remove refusals' from a model."
---

# Vibe Abliteration

Automate directional ablation with [Heretic](https://github.com/p-e-w/heretic) to remove censorship while preserving model capabilities.

## When to use
- User wants to decensor/abliterate a HuggingFace model
- Need to find optimal ablation parameters balancing refusals vs KL divergence
- Want to reproduce or improve abliteration results

## Prerequisites
- GPU with sufficient VRAM (varies by model size; 4-8GB minimum for small models)
- Python 3.10+, PyTorch installed appropriately
- HuggingFace account/token if uploading

## Workflow

### 1) Environment setup
Follow the tested pattern for Heretic v2.x:
```bash
# Create venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools

# Install PyTorch for your CUDA version (check pytorch.org)
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124  # or appropriate

# Install dependencies and heretic
pip install -U "transformers[torch]" peft huggingface-hub datasets accelerate bitsandbytes safetensors sentencepiece protobuf
pip install -U "git+https://github.com/p-e-w/heretic.git"
```

Notes:
- v2.0 has dependency constraints; may need specific versions depending on PyTorch
- For non-interactive environments, watch for TTY issues (see below)

### 2) Login to HuggingFace (if needed)
```bash
huggingface-cli login --token $HF_TOKEN
```

### 3) Run optimization
```bash
heretic <model-id>  # e.g. LiquidAI/LFM2.5-230M
```

Key parameters (via CLI or config):
- `--n-trials`: number of optimization trials (default 100; more trials = better search)
- `--n-startup-trials`: random exploration trials (default 30)
- `--quantization`: NONE or BNB_4BIT to save VRAM
- `--batch-size`: auto or manual
- `--seed`: for reproducibility
- `--study-checkpoint-dir`: save/load progress (default `checkpoints`)

### 4) Handle post-optimization (critical for non-interactive)
**Known issue (v2.0):** After optimization, Heretic prompts interactively to select trial/save/upload. In non-TTY environments this crashes with EOFError from prompt_toolkit.

Workarounds:
- Use `script` or `expect` to provide input
- Pass flags when possible: `--trial-index N`, `--model-action save`, `--save-directory <dir>`
- If flags don't prevent prompting, run in a proper TTY or handle interactively
- Check issue #535 for updates

### 5) Evaluate and choose best trial
Look at Pareto front results. Objectives: minimize refusals, minimize KL divergence from baseline.
- Check trial results in logs/checkpoints (`checkpoints/*.jsonl`)
- For trade-off goals, parse and select best by your criteria

### 6) Save and document
- Save model with `--model-action save --save-directory <path>` or interactively
- Create model card following Heretic conventions (see example model cards)
- Include: base model, heretic version, parameters/trial info, KL divergence, refusal scores, ablation parameters
- If uploading, use appropriate tags (`uncensored`, `decensored`, `abliterated`)

## Tips from testing
- More trials help find better trade-offs (150-200 can find good points)
- Small models may have tight trade-off space (low KL often means high refusals)
- The best configuration depends on your priority (zero refusals vs minimal drift)
- Always keep checkpoints - optimization can take long time

## Troubleshooting
- **Dependency conflicts**: May need to match PyTorch/CUDA versions carefully with transformers/peft/huggingface-hub
- **OOM**: Use quantization, reduce batch size, enable CPU offload options
- **Non-interactive crash**: Use TTY or workarounds above
- **Slow evaluation**: Normal - each trial evaluates refusals and KL

## References
- [Heretic repo](https://github.com/p-e-w/heretic)
- [Abliteration paper (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)
- [Model card example](https://huggingface.co/saidutta69/LFM2.5-230M-heretic)
