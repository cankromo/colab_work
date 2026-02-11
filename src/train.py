import argparse
import os

import supersuit as ss
from stable_baselines3 import PPO

from callbacks import OutcomeStatsCallback
from custom_environment import CustomEnvironment

os.environ["SDL_VIDEODRIVER"] = "dummy"


def train(args):
    env = CustomEnvironment(
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        render_mode=None,
        max_steps=args.max_steps,
        seed=args.seed
    )

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=args.num_vec_envs,
        num_cpus=args.num_cpus,
        base_class="stable_baselines3",
    )

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

    callback = OutcomeStatsCallback(print_every=args.print_every, window=args.window)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)

    env.close()
    print("Training complete, model saved to:", args.save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_vec_envs", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--save_path", type=str, default="models/ppo_guards.zip")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--print_every", type=int, default=5000)
    p.add_argument("--window", type=int, default=200)

    args = p.parse_args()
    train(args)