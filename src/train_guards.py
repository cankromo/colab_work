import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import supersuit as ss
from stable_baselines3 import PPO

from .custom_environment import CustomEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_vec_envs", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--save_path", type=str, default="models/guard_model.zip")
    p.add_argument("--seed", type=int, default=0)

    # prisoner sabit kalsın (heuristic ile başlamak en stabil)
    p.add_argument("--prisoner_use_heuristic", action="store_true", default=True)

    args = p.parse_args()

    env = CustomEnvironment(
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        max_steps=args.max_steps,
        training_side="guards",
        prisoner_use_heuristic=args.prisoner_use_heuristic,
        render_mode=None,
        seed=args.seed,
    )

    # SB3 uyumu
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, args.num_vec_envs, num_cpus=args.num_cpus, base_class="stable_baselines3")

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

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    env.close()

    print("✅ Saved guard model:", args.save_path)


if __name__ == "__main__":
    main()