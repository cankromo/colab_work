# Project Overview

This project appears to be a machine learning or reinforcement learning project, likely involving agents (guards and prisoners) and grid environments, given the file structure.

Key components include:
- `src/`: Source code for the project, including environment definitions, training scripts, evaluation metrics, and utilities.
- `notebooks/`: Jupyter notebooks for experiments and analysis (e.g., `can.ipynb`).
- `eval/`: Contains evaluation results, such as `rollout_returns.csv`.
- `plots/`: Stores various plots and visualizations generated during training and evaluation (e.g., episode length, outcomes, returns, loss, rewards, PPO diagnostics).
- `results/`: Contains trained models and experiment logs, organized by `exp` (experiment) and `grids` (grid environments).
  - `exp1`, `exp2`, `exp3`: Different experiment runs, each potentially containing `model.zip`, `guard_model.zip`, `prisoner_model.zip`, and log files.
  - `grids`, `grids_scaled`: Results specifically related to grid environments, further organized by `grid_100` which contains evaluation data, models, logs, and play samples (e.g., `play_both_ep1.mp4`).
- `requirements.txt`: Specifies the Python dependencies for the project.

## Directory Structure

- `eval/`
  - `rollout_returns.csv`: Contains data related to evaluation rollouts.
- `notebooks/`
  - `can.ipynb`: A Jupyter notebook for interactive development or analysis.
- `plots/`
  - Various `.png` files for visualizing training and evaluation metrics.
- `results/`
  - `exp1/`, `exp2/`, `exp3/`: Experiment-specific results.
  - `grids/`, `grids_scaled/`: Grid environment-specific results.
    - Contains models (`guard_model.zip`, `prisoner_model.zip`), logs, and visual outputs.
- `src/`
  - `callbacks_metrics.py`, `callbacks.py`: Callbacks for training.
  - `compare_models.py`: Script for comparing different models.
  - `custom_environment.py`: Definition of the custom environment.
  - `eval_rollout_rewards.py`: Script for evaluating rollout rewards.
  - `__init__.py`, `__pycache__/`: Python package structure.
  - `play_both.py`: Script for playing both agents.
  - `plot_metrics.py`: Utilities for plotting metrics.
  - `run_grid_experiments.py`: Script to run grid experiments.
  - `train_guards.py`, `train_prisoner.py`: Scripts for training guard and prisoner agents.
- `requirements.txt`: Project dependencies.

## Getting Started

To set up the project locally:

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run experiments or analysis**:
    - Explore the notebooks in `notebooks/`.
    - Run training scripts from `src/` (e.g., `python src/train_guards.py`).
    - Review evaluation results in `eval/` and `results/`.
    - View generated plots in `plots/`.

## Authors

[Your Name/Team Name]

## License

[Specify License, e.g., MIT, Apache 2.0]
