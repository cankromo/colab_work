import os
import sys
import argparse
import subprocess


def run_cmd(cmd, env=None, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


EXPECTED_PLOTS = [
    "train_reward_guards.png",
    "train_reward_prisoner.png",
    "train_loss_guards.png",
    "train_loss_prisoner.png",
    "eval_10k_returns.png",
    "eval_10k_winrate.png",
    "eval_10k_outcomes.png",
    "eval_10k_episode_length.png",
    "train_reward_guards_smoothed.png",
    "train_reward_prisoner_smoothed.png",
    "ppo_diagnostics_guards.png",
    "ppo_diagnostics_prisoner.png",
    "train_eval_link_guards.png",
    "train_eval_link_prisoner.png",
]


def auto_params_for_grid(grid_size):
    # Tuned for large-grid runs so evaluation is not dominated by timeouts.
    if grid_size <= 100:
        return {"max_steps": 400, "timesteps": 400_000, "episodes": 5_000}
    if grid_size <= 500:
        return {"max_steps": 1_500, "timesteps": 800_000, "episodes": 3_000}
    return {"max_steps": 3_000, "timesteps": 1_200_000, "episodes": 2_000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_sizes", type=str, default="100,500,1000")
    p.add_argument("--base_out", type=str, default="results/grids_scaled")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--reward_mode", type=str, default="legacy", choices=["legacy", "dynamic"])
    p.add_argument("--guard_time_penalty", type=float, default=None)
    p.add_argument("--prisoner_time_penalty", type=float, default=None)
    p.add_argument("--time_pressure_lambda", type=float, default=1.0)
    p.add_argument("--guard_timeout_penalty", type=float, default=None)
    p.add_argument("--prisoner_timeout_penalty", type=float, default=None)
    p.add_argument("--num_vec_envs", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--auto_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale max_steps/timesteps/episodes by grid size.",
    )
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    args = p.parse_args()

    sizes = [int(s.strip()) for s in args.grid_sizes.split(",") if s.strip()]
    if not sizes:
        raise SystemExit("No grid sizes provided")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_base = os.environ.copy()
    existing_pp = env_base.get("PYTHONPATH", "")
    env_base["PYTHONPATH"] = (
        project_root if not existing_pp else f"{project_root}{os.pathsep}{existing_pp}"
    )

    for gs in sizes:
        if args.auto_scale:
            cfg = auto_params_for_grid(gs)
            max_steps = cfg["max_steps"]
            timesteps = cfg["timesteps"]
            episodes = cfg["episodes"]
        else:
            max_steps = args.max_steps
            timesteps = args.timesteps
            episodes = args.episodes

        print(
            f"Config grid_size={gs}: max_steps={max_steps}, "
            f"timesteps={timesteps}, episodes={episodes}"
        )

        grid_dir = os.path.join(args.base_out, f"grid_{gs}")
        log_dir = os.path.join(grid_dir, "log")
        eval_csv = os.path.join(grid_dir, "eval", "rollout_returns.csv")
        plot_dir = os.path.join(grid_dir, "plots")
        guard_model = os.path.join(grid_dir, "guard_model.zip")
        prisoner_model = os.path.join(grid_dir, "prisoner_model.zip")

        os.makedirs(grid_dir, exist_ok=True)

        if not args.skip_train:
            guard_train_cmd = [
                sys.executable,
                "-m",
                "src.train_guards",
                "--grid_size",
                str(gs),
                "--num_guards",
                str(args.num_guards),
                "--max_steps",
                str(max_steps),
                "--reward_mode",
                args.reward_mode,
                "--timesteps",
                str(timesteps),
                "--num_vec_envs",
                str(args.num_vec_envs),
                "--num_cpus",
                str(args.num_cpus),
                "--seed",
                str(args.seed),
                "--save_path",
                guard_model,
                "--log_dir",
                log_dir,
                "--run_name",
                "guards",
            ]
            if args.guard_time_penalty is not None:
                guard_train_cmd += ["--guard_time_penalty", str(args.guard_time_penalty)]
            if args.prisoner_time_penalty is not None:
                guard_train_cmd += ["--prisoner_time_penalty", str(args.prisoner_time_penalty)]
            guard_train_cmd += ["--time_pressure_lambda", str(args.time_pressure_lambda)]
            if args.guard_timeout_penalty is not None:
                guard_train_cmd += ["--guard_timeout_penalty", str(args.guard_timeout_penalty)]
            if args.prisoner_timeout_penalty is not None:
                guard_train_cmd += ["--prisoner_timeout_penalty", str(args.prisoner_timeout_penalty)]
            run_cmd(
                guard_train_cmd,
                env=env_base,
                cwd=project_root,
            )

            prisoner_train_cmd = [
                sys.executable,
                "-m",
                "src.train_prisoner",
                "--grid_size",
                str(gs),
                "--num_guards",
                str(args.num_guards),
                "--max_steps",
                str(max_steps),
                "--reward_mode",
                args.reward_mode,
                "--timesteps",
                str(timesteps),
                "--num_vec_envs",
                str(args.num_vec_envs),
                "--num_cpus",
                str(args.num_cpus),
                "--seed",
                str(args.seed),
                "--guard_model_path",
                guard_model,
                "--save_path",
                prisoner_model,
                "--log_dir",
                log_dir,
                "--run_name",
                "prisoner",
            ]
            if args.guard_time_penalty is not None:
                prisoner_train_cmd += ["--guard_time_penalty", str(args.guard_time_penalty)]
            if args.prisoner_time_penalty is not None:
                prisoner_train_cmd += ["--prisoner_time_penalty", str(args.prisoner_time_penalty)]
            prisoner_train_cmd += ["--time_pressure_lambda", str(args.time_pressure_lambda)]
            if args.guard_timeout_penalty is not None:
                prisoner_train_cmd += ["--guard_timeout_penalty", str(args.guard_timeout_penalty)]
            if args.prisoner_timeout_penalty is not None:
                prisoner_train_cmd += ["--prisoner_timeout_penalty", str(args.prisoner_timeout_penalty)]
            run_cmd(
                prisoner_train_cmd,
                env=env_base,
                cwd=project_root,
            )

        if not args.skip_eval:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.eval_rollout_rewards",
                    "--grid_size",
                    str(gs),
                    "--num_guards",
                    str(args.num_guards),
                    "--max_steps",
                    str(max_steps),
                    "--reward_mode",
                    args.reward_mode,
                    "--episodes",
                    str(episodes),
                    "--seed",
                    str(args.seed),
                    "--guard_model_path",
                    guard_model,
                    "--prisoner_model_path",
                    prisoner_model,
                    "--out_csv",
                    eval_csv,
                ]
                + (["--guard_time_penalty", str(args.guard_time_penalty)] if args.guard_time_penalty is not None else [])
                + (["--prisoner_time_penalty", str(args.prisoner_time_penalty)] if args.prisoner_time_penalty is not None else [])
                + ["--time_pressure_lambda", str(args.time_pressure_lambda)]
                + (["--guard_timeout_penalty", str(args.guard_timeout_penalty)] if args.guard_timeout_penalty is not None else [])
                + (["--prisoner_timeout_penalty", str(args.prisoner_timeout_penalty)] if args.prisoner_timeout_penalty is not None else []),
                env=env_base,
                cwd=project_root,
            )

        if not args.skip_plots:
            env = env_base.copy()
            env["PLOT_BASE_DIR"] = project_root
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.plot_metrics",
                    "--log_dir",
                    log_dir,
                    "--run_guards",
                    "guards",
                    "--run_prisoner",
                    "prisoner",
                    "--eval_csv",
                    eval_csv,
                    "--plot_dir",
                    plot_dir,
                ],
                env=env,
                cwd=project_root,
            )
            missing = []
            for name in EXPECTED_PLOTS:
                plot_path = os.path.join(project_root, plot_dir, name)
                if not os.path.exists(plot_path):
                    missing.append(plot_path)
            if missing:
                print("Missing plots:")
                for mp in missing:
                    print(" -", mp)
                raise RuntimeError(f"Plot generation incomplete for grid_size={gs}")

        print(f"Done grid_size={gs} -> {grid_dir}")


if __name__ == "__main__":
    main()
