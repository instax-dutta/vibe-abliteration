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

## Before you start: is it already published?

Optional shortcut, not a required step. If the user only needs a working
decensored model (not a specific parameter set or a fresh run), check
[Heretic Directory](https://hdir.sdad.pro) first — it indexes published
Heretic conversions and GGUF/MLX/AWQ quants, ranked by KL divergence and
refusal rate.

- Search the **original** model name (`Qwen3-4B`), not the uncensored one.
- Model page pattern: `https://hdir.sdad.pro/models/<org>/<name>/`
  (e.g. <https://hdir.sdad.pro/models/qwen/qwen3-4b/>) lists each
  conversion with its repo, formats, KL, refusals and downloads.
- Numbers there are publisher-reported and not re-measured; prefer rows
  with published metrics over inherited ones.
- If a conversion fits, hand the repo id to the user and stop. Run Heretic
  when no conversion exists, when the user wants better numbers, or when
  they want to control the trade-off themselves.

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
pip install -U heretic-llm
```

Notes:
- v2.0 has dependency constraints; may need specific versions depending on PyTorch
- For headless runs, set all post-optimization flags (see section 4)

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

### 4) Handle post-optimization
After optimization, Heretic asks which trial to use and what to do with the model. Set these flags to skip the prompts:

- `--trial-index N`: position in the sorted Pareto front, not the `[Trial N]` label printed in the menu
- `--model-action save` or `--model-action upload`
- `--save-directory <dir>` for save, or `--upload-repo-id <repo>` for upload
- `--export-strategy merge` or `--export-strategy adapter`
- `--upload-repo-private` and `--upload-reproducibility-information` for upload
- `--checkpoint-action continue` or `restart` when a journal already exists
- For upload, set `HF_TOKEN` or run `hf auth login` first; the token is deliberately not a CLI setting

Example headless save:
```bash
heretic <model-id> --trial-index 0 --model-action save --save-directory ./out --export-strategy merge
```

If a setting is left unset, Heretic prompts for it. With no TTY, that prompt raises `EOFError`, which is the expected result of a missing flag, not a bug to work around.

### 5) Evaluate and choose best trial
Objectives: minimize refusals, minimize KL divergence from baseline. Look at the Pareto front, not a single winner.

Summarize the journal without reloading the model:
```bash
python3 scripts/evaluate_trials.py checkpoints --json   # or omit --json for a table
```
It prints both trial numbers because they differ:
- `index` is Heretic's `[Trial N]` label.
- `trial_index` is the position in the sorted Pareto front, which is the number `--trial-index` expects.

Pass the `trial_index` of the point you want. A trial that is not on the Pareto front has no `trial_index` and cannot be selected headlessly.

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
- **`EOFError` in a headless run**: a required post-optimization flag is unset. Set the flags in section 4.
- **Slow evaluation**: Normal - each trial evaluates refusals and KL

## References
- [Heretic repo](https://github.com/p-e-w/heretic)
- [Abliteration paper (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)
- [Model card example](https://huggingface.co/saidutta69/LFM2.5-230M-heretic)

## Seeds and search strategy

### What is `--seed`?
The `--seed` parameter seeds all randomness (Python, NumPy, PyTorch, Optuna's TPE sampler). It controls:
- **Startup exploration**: Which random parameter combinations are tried in `--n-startup-trials` (default 30)
- **Guided search path**: How TPE proposes subsequent trials based on those starting points
- **Reproducibility**: Same seed + same model + same config = reproducible results

### Impact on results
- **Path selection**: Different seeds explore completely different regions of the parameter space. A good seed can find the "knee" of the Pareto curve quickly; a bad seed can miss it.
- **More trials vs better seed**: More trials help *along* a chosen path, but trying multiple seeds often beats doubling trials. For tight constraints (e.g., "<10 refusals with minimal KL"), running 2-3 seeds with ~50-100 trials each is often more effective than 1 seed with 300 trials.
- **Diminishing returns**: For models <1B, biggest gains come in first 50-150 trials per seed; going beyond often gives smaller improvements.

### Recommendations
```bash
# Try multiple seeds
heretic <model> --seed 42 --n-trials 150
heretic <model> --seed 12345 --n-trials 150
heretic <model> --seed 777 --n-trials 150
```

- **Keep checkpoints**: Use `--study-checkpoint-dir`. If resuming/extending, keep the same seed and directory. Start fresh directory + new seed for a different search path.
- **Optimize for the knee**: Don't chase 0/100 refusals at all costs - often the best real-world trade-off is just under 10 refusals with minimal KL. Always look at the Pareto front.
- **Compare across seeds**: Evaluate all trials from all seeds together to find the true best trade-off for your goal.

## Tested results (Heretic v2.0.0.dev0)

### EuroLLM-1.7B-Instruct (seed 42, 200 trials, base 4/100 refusals)
| Trial | KL | Refusals | Notes |
|-------|-----|----------|-------|
| 153 | 0.0119 | 3/100 | Small improvement |
| 154 | 0.0083 | 2/100 | Best tradeoff - preferred for upload |
| 174 | 0.0068 | 3/100 | Lower KL, slightly more refusals than 154 |
| 170 | 0.1939 | 0/100 | No refusals but high KL, not preferred |

Reproduce: `heretic --model utter-project/EuroLLM-1.7B-Instruct --seed 42 --n-trials 200`

### SmolLM2-135M-Instruct (150 trials)
| Trial | KL | Refusals | Notes |
|-------|-----|----------|-------|
| 135 | 0.0006 | 9/100 | Beats baseline with negligible KL |
| 68 | 0.0011 | 9/100 | Beats baseline with very low KL |
| 38 | 0.0124 | 4/100 | 60% reduction |
| 70 | 0.0174 | 4/100 | 60% reduction |

### LFM2.5-350M (150 trials)
58/150 trials under KL 0.02 but ~89/100 refusals - not viable under target constraints.

Full testing report: [gist](https://gist.github.com/instax-dutta/b456ff884c9f570434d964ca8e633297)

