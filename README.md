# vibe-abliteration

A robust, agent-friendly skill for automating model censorship removal (abliteration) with [Heretic](https://github.com/p-e-w/heretic). Finds optimal directional ablation parameters that minimize refusals while preserving model capabilities (co-minimizing refusals and KL divergence).

## What it does

- **Environment setup** - Installs correct dependencies for your GPU/CUDA
- **Automated optimization** - Runs Heretic's Optuna-based search to find best ablation parameters
- **Smart evaluation** - Balances refusal suppression vs KL drift from original model
- **Handles headless runs** - Set the post-optimization flags to save and upload without a TTY
- **Model export & documentation** - Saves models with proper Heretic-style model cards
- **Checkpoint management** - Resume interrupted runs

## Quick Start

Use with [Claude Code](https://docs.anthropic.com/claude/docs/claude-code), [OpenCode](https://github.com/opencode-ai/opencode), [Kilo Code](https://kilocode.ai/), [Cursor](https://www.cursor.com/), or any agent that supports skills.

### Installation
```bash
# Via skills.sh (coming soon)
npx skills add instax-dutta/vibe-abliteration

# Or manually copy to your skills directory
cp -r vibe-abliteration ~/.kilocode/skills/
# or ~/.claude/skills/, ~/.config/opencode/skills/, etc.
```

## Usage

Tell your agent: *"Abliterate LiquidAI/LFM2.5-230M with Heretic, minimize KL while getting refusals under 10/100"*

Or use the skill directly with the included scripts.

## Scripts

- `scripts/setup_env.sh` - Create venv and install dependencies
- `scripts/run_abliteration.sh` - Run optimization with sensible defaults
- `scripts/evaluate_trials.py` - Parse checkpoints and find best trials
- `scripts/save_model.sh` - Save best trial non-interactively
- `scripts/make_model_card.py` - Generate Heretic-style README/model card

## Related Skills

- [finetune-llm](https://github.com/instax-dutta/finetune-llm) - Hardware-aware LLM fine-tuning
- [market-validator](https://github.com/instax-dutta/market-validator) - Validate ideas with real user complaints
- [master-pitcher](https://github.com/instax-dutta/master-pitcher) - Pitch deck analysis
- [flash-compare](https://github.com/instax-dutta/flash-compare) - Product comparisons
- [google-code-review](https://github.com/instax-dutta/google-code-review) - Code review best practices
- [dox-init](https://github.com/instax-dutta/dox-init) - Initialize DOX documentation trees
- [darpa-audit](https://github.com/instax-dutta/darpa-audit) - Startup idea validation
- [brand-vibes](https://github.com/instax-dutta/brand-vibes) - Apply brand design languages
- [roadmap-tutor](https://github.com/instax-dutta/roadmap-tutor) - Learn roadmaps interactively
- [scroll-3d-world](https://github.com/instax-dutta/scroll-3d-world) - 3D scroll experiences
- [Only Skills You Need](https://github.com/instax-dutta/only-skills-you-need) - Curated skills stack

## Tips from Testing

From extensive testing on LFM2.5-230M (450+ trials):
- Heretic v2.0 can find good trade-offs; sometimes better than v1.4.0
- Headless save and upload need every post-optimization flag set (`--trial-index`, `--model-action`, `--export-strategy`, and the save or upload flags). If one is missing, Heretic prompts and fails without a TTY.
- Small models can have tight trade-off spaces between KL and refusals
- More trials (150-200) often find better Pareto-optimal solutions

## License

MIT
