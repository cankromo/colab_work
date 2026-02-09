"""
Compares the performance of multiple pre-trained SB3 agents in the Waterworld environment.
This script iterates through a list of model files, evaluates each one, and prints a performance summary.
"""
from __future__ import annotations

import os

from stable_baselines3 import PPO

from pettingzoo.sisl import waterworld_v4
from waterworld_train import eval_agent

# Set a dummy video driver for headless environments.
os.environ["SDL_VIDEODRIVER"] = "dummy"


def compare_models(env_fn, model_paths: list[str], num_games: int, **env_kwargs):
    """
    Evaluates multiple models and prints a comparison of their average rewards.

    Args:
        env_fn: The environment function from PettingZoo.
        model_paths: A list of file paths for the trained models to compare.
        num_games: The number of games to evaluate each model on.
        **env_kwargs: Additional arguments for the environment.
    """
    print(f"\n--- Starting Model Performance Comparison ---")
    print(f"Evaluating {len(model_paths)} models over {num_games} games each.")

    performance_results = {}

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found, skipping: {model_path}")
            continue

        print(f"\n--- Evaluating model: {model_path} ---")
        avg_reward = eval_agent(
            env_fn, num_games=num_games, model_path=model_path, **env_kwargs
        )
        performance_results[model_path] = avg_reward

    print("\n\n--- Performance Comparison Summary ---")
    # Sort results from best to worst average reward
    sorted_results = sorted(performance_results.items(), key=lambda item: item[1], reverse=True)
    for model_path, avg_reward in sorted_results:
        print(f"Model: {model_path:<30} | Average Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    # Define the environment and arguments.
    env_function = waterworld_v4
    environment_kwargs = {"n_pursuers": 2}

    # List of models you want to compare.
    models_to_compare = ["ppo_waterworld_default_lr.zip", "ppo_waterworld_low_lr.zip"]

    # Run the comparison.
    compare_models(env_function, models_to_compare, num_games=50, **environment_kwargs)