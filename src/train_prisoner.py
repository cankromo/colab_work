import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from .custom_environment import CustomEnvironment
from .callbacks_metrics import EpisodeReturnLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_vec_envs", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="legacy", choices=["legacy", "dynamic"])
    p.add_argument("--guard_time_penalty", type=float, default=None)
    p.add_argument("--prisoner_time_penalty", type=float, default=None)

    p.add_argument("--guard_model_path", type=str, default="models/guard_model.zip")
    p.add_argument("--save_path", type=str, default="models/prisoner_model.zip")

    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--run_name", type=str, default="prisoner")

    args = p.parse_args()

    env = CustomEnvironment(
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        max_steps=args.max_steps,
        training_side="prisoner",
        reward_mode=args.reward_mode,
        guard_model_path=args.guard_model_path,
        guard_time_penalty=args.guard_time_penalty,
        prisoner_time_penalty=args.prisoner_time_penalty,
        render_mode=None,
        seed=args.seed,
    )

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, args.num_vec_envs, num_cpus=args.num_cpus, base_class="stable_baselines3")

    # ✅ episodic return için şart
    env = VecMonitor(env)

    run_path = os.path.join(args.log_dir, args.run_name)
    new_logger = configure(run_path, ["stdout", "csv", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=run_path,
        
    )
    model.set_logger(new_logger)

    cb = EpisodeReturnLogger(log_dir=args.log_dir, run_name=args.run_name)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save(args.save_path)

    env.close()
    print("✅ Saved prisoner model:", args.save_path)
    print("✅ Logs at:", run_path)


if __name__ == "__main__":
    main()
