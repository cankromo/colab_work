# Guards vs Prisoner RL

Multi-agent reinforcement learning project using a grid-world pursuit/escape setting:
- Guards try to capture the prisoner.
- Prisoner tries to reach the escape cell.

The codebase supports training, evaluation, plotting, and grid-size experiment sweeps with both legacy and dynamic reward shaping.

## Tech Stack

- Python
- PettingZoo (parallel environment)
- SuperSuit (vectorization wrappers)
- Stable-Baselines3 PPO
- NumPy
- Matplotlib/Pandas (plotting)
- Pygame + ImageIO (render/video)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `src/custom_environment.py`: Core PettingZoo environment and reward logic.
- `src/train_guards.py`: Train guard policy (prisoner controlled by heuristic).
- `src/train_prisoner.py`: Train prisoner policy (guards controlled by fixed guard model).
- `src/eval_rollout_rewards.py`: Evaluate trained guard/prisoner models and export rollout CSV.
- `src/plot_metrics.py`: Generate training/eval plots from logs and evaluation CSV.
- `src/run_grid_experiments.py`: End-to-end pipeline for multiple grid sizes.
- `src/play_both.py`: Run both trained agents and save gameplay MP4.
- `src/compare_models.py`: Outcome-level benchmark (captured/escaped/timeout rates).
- `results/`, `plots/`, `eval/`: Experiment artifacts (committed in this repository).

## Reward Modes

`CustomEnvironment` supports:

- `legacy` (default): existing shaping behavior.
- `dynamic`: relative step-progress shaping to reduce scale bias on large grids.

Available via `--reward_mode` in:
- `src/train_guards.py`
- `src/train_prisoner.py`
- `src/eval_rollout_rewards.py`
- `src/run_grid_experiments.py`
- `src/play_both.py`
- `src/compare_models.py`

## Quick Start (Single Grid)

Run commands from repository root (`/content/colab_work`) with module mode (`python -m ...`).

1. Train guards:

```bash
python -m src.train_guards \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 400 \
  --timesteps 400000 \
  --reward_mode dynamic \
  --save_path results/manual/guard_model.zip \
  --log_dir results/manual/log \
  --run_name guards
```

2. Train prisoner against trained guard model:

```bash
python -m src.train_prisoner \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 400 \
  --timesteps 400000 \
  --reward_mode dynamic \
  --guard_model_path results/manual/guard_model.zip \
  --save_path results/manual/prisoner_model.zip \
  --log_dir results/manual/log \
  --run_name prisoner
```

3. Evaluate rollouts:

```bash
python -m src.eval_rollout_rewards \
  --episodes 5000 \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 400 \
  --reward_mode dynamic \
  --guard_model_path results/manual/guard_model.zip \
  --prisoner_model_path results/manual/prisoner_model.zip \
  --out_csv results/manual/eval/rollout_returns.csv
```

4. Generate plots:

```bash
python -m src.plot_metrics \
  --log_dir results/manual/log \
  --run_guards guards \
  --run_prisoner prisoner \
  --eval_csv results/manual/eval/rollout_returns.csv \
  --plot_dir results/manual/plots
```

## Grid Sweep Pipeline (100/500/1000)

End-to-end training + evaluation + plots:

```bash
python -m src.run_grid_experiments \
  --grid_sizes 100,500,1000 \
  --base_out results/grids_scaled \
  --reward_mode dynamic \
  --auto_scale \
  --num_guards 2 \
  --seed 0
```

Useful flags:
- `--skip_train`: only eval/plots
- `--skip_eval`: only training/plots
- `--skip_plots`: only training/eval
- `--no-auto_scale`: use manual `--max_steps`, `--timesteps`, `--episodes`

## Play and Compare

Generate gameplay video:

```bash
python -m src.play_both \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 400 \
  --reward_mode dynamic \
  --guard_model_path results/manual/guard_model.zip \
  --prisoner_model_path results/manual/prisoner_model.zip \
  --video_path results/manual/play_samples/play_both.mp4
```

Compute win/timeout statistics:

```bash
python -m src.compare_models \
  --episodes 1000 \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 400 \
  --reward_mode dynamic \
  --guard_model_path results/manual/guard_model.zip \
  --prisoner_model_path results/manual/prisoner_model.zip
```

## Output Artifacts

Typical outputs per run:
- Models: `guard_model.zip`, `prisoner_model.zip`
- Logs: `progress.csv`, TensorBoard `events.out.tfevents.*`, `episode_returns.csv`
- Eval: `rollout_returns.csv`
- Plots: reward/loss/eval diagnostics PNG set
- Optional: gameplay MP4

## Notes

- Use `python -m src.<script>` instead of `python src/<script>.py` (scripts use package-relative imports).
- `.gitignore` is currently focused on Python caches, notebook checkpoints, and `.env` variants.
