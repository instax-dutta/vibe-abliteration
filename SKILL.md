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

### Check the model can be abliterated before spending GPU time
Heretic supports most dense models, many multimodal models, several MoE
architectures, and many modern hybrids. It locates layers at
`model.model.language_model.layers` (multimodal) or `model.model.layers`, then
modifies whichever of these exist:

| Component | Modules it accepts |
|-----------|--------------------|
| `attn.o_proj` | `self_attn.o_proj`, `linear_attn.out_proj`, `self_attn.o_b_proj`, `conv.out_proj` |
| `mlp.down_proj` | `mlp.down_proj`, MoE expert `down_proj`/`w2`, shared-expert projections, `feed_forward.w2`, `shared_mlp.output_linear` |

So MoE experts, hybrid linear-attention layers and LFM conv blocks are covered.
What is not covered is anything without a transformer-style `.layers` stack, or
whose attention/MLP output projection is named differently.

Two failure signatures mean "unsupported", not "bug to work around":
- missing `.layers` attribute on the loaded model
- `No modifiable modules found in layer`

Gated/private repos need a token. Prefer `safetensors` checkpoints.

## Workflow

### 1) Environment setup
Follow the tested pattern for Heretic v2.x:
```bash
# Create venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools

# Install PyTorch for your CUDA version (check pytorch.org)
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124  # or appropriate

# Install Heretic; it pins its own dependencies (transformers, peft, datasets, lm-eval, optuna, ...)
pip install -U heretic-llm
```

Notes:
- Install PyTorch first so Heretic does not pull a CPU-only build; `heretic-llm` accepts whatever torch it finds.
- `bitsandbytes` arrives with Heretic and is what `--quantization bnb_4bit` needs.
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
- `--model`: model id (also accepted positionally)

List settings (`scorers`, `modifiers`, `benchmarks`) cannot be expressed as
flags. Put them in `config.toml` in the working directory, or use `HERETIC_*`
environment variables. See the reasoning layer below for what to put there.

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

### 6) Verify, save and document
Verify before publishing, because refusal counts are keyword-matched and the
export is a different artifact from the trial:
- Load the saved model and run a handful of prompts the baseline refused. They
  should now answer without degenerate output.
- Spot-check normal prompts for repetition, broken formatting or lost reasoning.
- If capability matters, run one benchmark (see below) rather than trusting KL
  alone.

Then:
- Save model with `--model-action save --save-directory <path>` or interactively
- Create model card following Heretic conventions (see example model cards)
- Include: base model, heretic version, parameters/trial info, KL divergence, refusal scores, ablation parameters
- If uploading, use appropriate tags (`uncensored`, `decensored`, `abliterated`)

## Reasoning layer

Heretic optimizes two proxies: keyword-matched refusals and KL divergence from
the original. Whether the result is good depends on the decisions below, so
make them deliberately rather than accepting defaults.

### What one trial actually is

Each trial samples an ablation profile, applies it, then measures the objectives:

- **Components.** Only two names are modified: `attn.o_proj` (attention output)
  and `mlp.down_proj` (MLP down projection). MoE experts, shared experts,
  hybrid linear attention, LFM conv blocks and DeepSeek `o_b_proj` all map onto
  these names.
- **Refusal directions.** Per layer, the mean residual over harmful prompts
  minus the mean over harmless prompts, normalized. With
  `orthogonalize_direction` the component along the harmless direction is
  removed first (projected abliteration). Both datasets are configurable.
- **`direction_scope`.** `global` uses one refusal direction, interpolated
  between adjacent layers. `per layer` uses each layer's own direction, which
  is more targeted and noisier. Both use the same precomputed per-layer
  residual directions, so neither costs extra compute.
- **`direction_index`.** Sampled in `0.4` to `0.9` x last layer index. The
  fractional part is the interpolation weight between two adjacent layer
  directions. Ignored when scope is `per layer`.
- **Weight profile per component.** Piecewise linear around a peak:
  - `max_weight`: `0.8`-`1.5` for attention; `-0.25`-`1.5` for MLP, clamped
    at `0` so the MLP can be disabled completely.
  - `max_weight_position`: `0.6`-`1.0` x last layer index.
  - `min_weight`: `0`-`1`, expressed as a fraction of `max_weight`.
  - `min_weight_distance`: `1.0` up to about `0.6` x layer count. Layers
    farther than this from the peak get no ablation at all.

Read trial parameters with that in mind:

- `mlp.down_proj.max_weight = 0` is normal, often optimal. MLP ablation rarely
  removes refusals and usually costs intelligence; attention carries most of
  the effect. Do not treat a disabled MLP as a failed trial.
- A peak past the midpoint of the stack is expected, matching Arditi et al.
  (2024).
- Untouched layers near the ends are the `min_weight_distance` cutoff working
  as designed, not gaps the optimizer failed to fill.

These values are not CLI settings. The optimizer owns them, and pinning a
specific profile requires writing a modifier plugin. The levers you really
have are the objective set, the datasets, the modifier settings, the trial
budget and the seed.

### Levers worth changing

`config.toml` in the working directory accepts the list settings that flags
cannot express:

```toml
# Optional: score a benchmark every trial without making it an objective.
[[scorers]]
plugin = "heretic.scorers.benchmark_score.BenchmarkScore"
optimization = "none"

[scorer.BenchmarkScore]
task = "piqa"
metric = "acc_norm,none"
score_name = "PIQA acc_norm"

# Ablation settings, applied to the built-in modifier.
[modifier.Abliteration]
orthogonalize_direction = true     # projected abliteration (default)
row_normalization = "full"         # magnitude-preserving (MPOA, default)
full_normalization_lora_rank = 3
winsorization_quantile = 0.95      # default 1.0 (off); helps "massive activations"

# Post-selection validation tasks (interactive menu only).
[[benchmarks]]
task = "piqa"
name = "PIQA"
description = "Physical interaction question answering"
```

Notes on these:
- `optimization = "none"` computes a score without adding it to the Pareto
  front. Use it when you want visibility but not another objective.
- `good_prompts` and `bad_prompts` under `[modifier.Abliteration]` redefine
  what counts as a refusal. The defaults are `mlabonne/harmless_alpaca` and
  `mlabonne/harmful_behaviors`, 400 rows each. Change them only deliberately.
- External plugin paths disable Heretic's reproducibility metadata on upload.

### When KL is not enough

- KL divergence is cheap and runs every trial. Use it to search.
- `BenchmarkScore` runs lm-eval on every trial. Accurate, but it multiplies
  per-trial cost; enable it when the decision hinges on capability, not by
  default.
- `[[benchmarks]]` run once on the chosen model, from the interactive menu, and
  are not reachable in a headless run. For a headless pipeline, validate the
  export separately with lm-eval or by chatting with it.
- Refusal counts come from keyword matching on harmful prompts and are
  publisher-reported when you read them from a model card. Treat both numbers
  as proxies and verify the behavior you actually care about.

### Budget

There is no time setting, only trial counts. Measure the first few trials, then
`n_trials = budget_seconds / seconds_per_trial`.

Extending a run is interactive only: `--checkpoint-action continue` reopens the
finished study, then the menu offers "Run additional trials" with
`n_additional_trials`. A headless run cannot extend, so plan the trial count up
front. For a headless pipeline, start a separate run with a new seed and its own
checkpoint directory rather than deleting history.

### Stop policy

- Target the knee of the Pareto front, not 0/100 refusals. For most models a
  point just under 10/100 refusals at the lowest reachable KL is the useful one.
- Gains flatten after roughly 50-150 trials per seed for sub-1B models. When
  that happens, a new seed or different datasets beats more trials.
- If the entire front sits at high refusals for low KL, the direction does not
  fit the model. More trials will not fix it; change the datasets or the
  modifier settings.

## Tips from testing
- More trials help until the front flattens; for sub-1B models that is often 50-150 per seed, and a new seed then beats more trials
- Small models may have tight trade-off space (low KL often means high refusals)
- The best configuration depends on your priority (zero refusals vs minimal drift)
- Always keep checkpoints - optimization can take long time

## Troubleshooting
- **Dependency conflicts**: Install PyTorch first, then let `heretic-llm` pin the rest. Do not force a different transformers major version.
- **OOM**: Use quantization, reduce batch size, enable CPU offload options
- **`EOFError` in a headless run**: a required post-optimization flag is unset. Set the flags in section 4.
- **`No modifiable modules found in layer`**: the architecture has no supported attention/MLP output projection. Not a bug; the model cannot be abliterated by Heretic.
- **Whole front stuck at high refusals**: the refusal direction does not fit the model. Change `good_prompts`/`bad_prompts` or `system_prompt`; adding trials will not help.
- **Slow evaluation**: Normal - each trial evaluates refusals and KL; a `BenchmarkScore` scorer multiplies this

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

