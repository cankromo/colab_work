import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
from typing import List
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from .custom_environment import CustomEnvironment
from .callbacks_metrics import EpisodeReturnLogger

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONTENT_DIR = os.path.dirname(BASE_DIR)


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    if p.startswith("colab_work/"):
        return os.path.join(CONTENT_DIR, p)
    return os.path.join(BASE_DIR, p)


def parse_float_list(v: str) -> List[float]:
    vals = [x.strip() for x in str(v).split(",") if x.strip()]
    if not vals:
        raise ValueError("curriculum_escape_probs must contain at least one value")
    out = [float(x) for x in vals]
    for p in out:
        if p < 0.0 or p > 1.0:
            raise ValueError("each curriculum escape probability must be in [0, 1]")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=40)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--num_vec_envs", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--capture_radius", type=float, default=2.0)
    p.add_argument("--escape_radius", type=float, default=2.0)
    p.add_argument("--guard_approach_reward_scale", type=float, default=8.0)
    p.add_argument("--guard_escape_penalty_lambda", type=float, default=0.01)
    p.add_argument("--guard_time_penalty", type=float, default=0.001)
    p.add_argument("--curriculum_escape_probs", type=str, default="0.8,0.6,0.5")
    p.add_argument("--save_path", type=str, default="colab_work/results/exp5/models/guard_model.zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", type=str, default="colab_work/results/exp5/log")
    p.add_argument("--run_name", type=str, default="guards")

    args = p.parse_args()

    save_path = resolve_path(args.save_path)
    log_dir = resolve_path(args.log_dir)
    curriculum_probs = parse_float_list(args.curriculum_escape_probs)

    def build_env(prisoner_escape_prob: float):
        env_local = CustomEnvironment(
            grid_size=args.grid_size,
            num_guards=args.num_guards,
            max_steps=args.max_steps,
            training_side="guards",
            prisoner_use_heuristic=True,
            prisoner_escape_prob=prisoner_escape_prob,
            capture_radius=args.capture_radius,
            escape_radius=args.escape_radius,
            guard_approach_reward_scale=args.guard_approach_reward_scale,
            guard_escape_penalty_lambda=args.guard_escape_penalty_lambda,
            guard_time_penalty=args.guard_time_penalty,
            render_mode=None,
            seed=args.seed,
        )

        env_local = ss.black_death_v3(env_local)
        env_local = ss.pettingzoo_env_to_vec_env_v1(env_local)
        env_local = ss.concat_vec_envs_v1(
            env_local,
            args.num_vec_envs,
            num_cpus=args.num_cpus,
            base_class="stable_baselines3",
        )
        return VecMonitor(env_local)

    env = build_env(curriculum_probs[0])

    # ✅ SB3 logger: CSV + TensorBoard
    run_path = os.path.join(log_dir, args.run_name)
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

    cb = EpisodeReturnLogger(log_dir=log_dir, run_name=args.run_name)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n_stages = len(curriculum_probs)
    base = args.timesteps // n_stages
    rem = args.timesteps % n_stages
    stage_steps = [base + (1 if i < rem else 0) for i in range(n_stages)]

    for i, prob in enumerate(curriculum_probs):
        if i > 0:
            new_env = build_env(prob)
            model.set_env(new_env)
            env.close()
            env = new_env
        print(f"=== Curriculum stage {i+1}/{n_stages}: prisoner_escape_prob={prob}, timesteps={stage_steps[i]} ===")
        model.learn(
            total_timesteps=stage_steps[i],
            callback=cb,
            reset_num_timesteps=(i == 0),
        )

    model.save(save_path)

    env.close()
    print("✅ Saved guard model:", save_path)
    print("✅ Logs at:", run_path)


if __name__ == "__main__":
    main()
