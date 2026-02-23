## Guards vs Prisoner (Multi‑Agent RL)

This project trains and evaluates two competing policies in a continuous 2D grid:

- **Guards** try to **capture** the prisoner.
- **Prisoner** tries to **reach the escape point**.

It uses a custom PettingZoo `ParallelEnv` plus Stable‑Baselines3 PPO. The repo includes
training scripts, evaluation scripts, and plotting utilities, along with several
experiment artifacts in `results/` and `plots/`.

---

## What We Tried (Current Results Snapshot)

The `eval/rollout_returns.csv` file contains evaluation rollouts for a trained guard
and prisoner model. A quick summary of that file:

- **Episodes**: 10,000
- **Outcomes**: 7,213 captured, 2,787 escaped
- **Mean episode length**: 2.7738
- **Mean return** (both sides): 0.4426

These are produced by `src/eval_rollout_rewards.py` and are visualized in
`plots/` and `results/*/plots/`. The plots include win‑rate curves, returns,
outcome counts, and PPO diagnostics.

---

## Repository Structure

- `src/custom_environment.py`: PettingZoo environment (continuous 2D, capture/escape/timeout)
- `src/train_guards.py`: trains guards vs heuristic prisoner (curriculum on escape behavior)
- `src/train_prisoner.py`: trains prisoner vs fixed guard model
- `src/eval_rollout_rewards.py`: evaluates trained models and writes `eval/rollout_returns.csv`
- `src/compare_models.py`: compares two models (or model vs heuristic)
- `src/plot_metrics.py`: plotting utilities for training and evaluation metrics
- `src/play_both.py`: renders a match and optionally saves MP4
- `results/`: experiments (models, logs, plots)
- `plots/`: summary plots
- `eval/`: evaluation CSVs

---

## Environment Details

**State**
- Continuous 2D positions and velocities for prisoner and guards.
- Escape point fixed per episode.

**Actions**
- Each agent outputs a 2‑D acceleration vector in `[-1, 1]`.

**Terminal conditions**
- **Captured**: prisoner is within `capture_radius` of any guard.
- **Escaped**: prisoner is within `escape_radius` of escape point.
- **Timeout**: episode reaches `max_steps`.

**Reward shaping**
- Guards: reward for reducing total guard‑to‑prisoner distance, penalty for escape progress, small time penalty.
- Prisoner: reward for reducing distance to escape, penalty for being close to guards, small time penalty.

---

## Setup

Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Training

### Train Guards (curriculum vs heuristic prisoner)

```bash
python -m src.train_guards \
  --grid_size 40 \
  --num_guards 2 \
  --max_steps 150 \
  --timesteps 300000 \
  --curriculum_escape_probs 0.8,0.6,0.5 \
  --save_path colab_work/results/exp5/models/guard_model.zip \
  --log_dir colab_work/results/exp5/log \
  --run_name guards
```

### Train Prisoner (vs fixed guard model)

```bash
python -m src.train_prisoner \
  --grid_size 100 \
  --num_guards 2 \
  --max_steps 100 \
  --timesteps 200000 \
  --guard_model_path colab_work/results/exp5/models/guard_model.zip \
  --save_path colab_work/results/exp5/models/prisoner_model.zip \
  --log_dir colab_work/results/exp5/log \
  --run_name prisoner
```

---

## Evaluation

Evaluate both trained models and write rollout returns:

```bash
python -m src.eval_rollout_rewards \
  --episodes 10000 \
  --guard_model_path colab_work/results/exp5/models/guard_model.zip \
  --prisoner_model_path colab_work/results/exp5/models/prisoner_model.zip \
  --out_csv colab_work/eval/rollout_returns.csv
```

Compare guard vs prisoner models (or model vs heuristic):

```bash
python -m src.compare_models \
  --episodes 1000 \
  --guard_model_path results/exp5/models/guard_model.zip \
  --prisoner_model_path results/exp5/models/prisoner_model.zip
```

---

## Plotting

Create plots from logs + evaluation CSV:

```bash
python -m src.plot_metrics \
  --exp_name exp7 \
  --results_dir results
```

This generates train curves, PPO diagnostics, win rates, and evaluation outcome plots.

---

## Rendering / Video

Render a match and save a video:

```bash
python -m src.play_both \
  --guard_model_path colab_work/results/exp5/models/guard_model.zip \
  --prisoner_model_path colab_work/results/exp5/models/prisoner_model.zip \
  --video_path colab_work/results/exp5/videos/play_both.mp4
```

Note: MP4s are ignored by `.gitignore` to keep repo size manageable.

---

## Notes

- Many scripts accept Colab‑style paths using the `colab_work/` prefix.
- `results/` already contains multiple experiments (`exp1`…`exp10`) with models and plots.
- If you want a clean experiment run, create a new `results/expX` directory and point `--log_dir`, `--save_path`, and `--out_csv` accordingly.
