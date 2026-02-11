import argparse
import os

import supersuit as ss
from stable_baselines3 import PPO

from callbacks import OutcomeStatsCallback
from custom_environment import CustomEnvironment

os.environ["SDL_VIDEODRIVER"] = "dummy"


def train(args):
    """Train the PPO model."""
    # 1️⃣ Create the environment
    env = CustomEnvironment(
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        render_mode=None
    )

    # 2️⃣ Apply PettingZoo wrappers
    env = ss.black_death_v3(env)

    # 3️⃣ Convert to a SB3 VecEnv
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 4️⃣ Create parallel environments
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=args.num_vec_envs,
        num_cpus=args.num_cpus,
        base_class="stable_baselines3"
    )

    # 5️⃣ Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # 6️⃣ Train the model
    callback = OutcomeStatsCallback(print_every=5000, window=100)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)

    env.close()
    print("Training complete, model saved to:", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for custom environment.")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid.")
    parser.add_argument("--num_guards", type=int, default=2, help="Number of guards.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps for training.")
    parser.add_argument("--num_vec_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use.")
    parser.add_argument("--save_path", type=str, default="models/ppo_guards.zip", help="Path to save the trained model.")

    args = parser.parse_args()
    train(args)

