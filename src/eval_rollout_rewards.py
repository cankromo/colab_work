import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import csv
import numpy as np
from stable_baselines3 import PPO

from .custom_environment import CustomEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="legacy", choices=["legacy", "dynamic"])

    p.add_argument("--guard_model_path", type=str, default="models/guard_model.zip")
    p.add_argument("--prisoner_model_path", type=str, default="models/prisoner_model.zip")

    p.add_argument("--out_csv", type=str, default="eval/rollout_returns.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    guard_model = PPO.load(args.guard_model_path)
    prisoner_model = PPO.load(args.prisoner_model_path)

    rng = np.random.default_rng(args.seed)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return_prisoner", "return_guards_mean", "outcome", "length"])

        for ep in range(1, args.episodes + 1):
            env = CustomEnvironment(
                render_mode=None,
                grid_size=args.grid_size,
                num_guards=args.num_guards,
                max_steps=args.max_steps,
                training_side="play",
                reward_mode=args.reward_mode,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            obs, infos = env.reset()

            ep_ret_p = 0.0
            ep_ret_g = 0.0
            steps = 0
            last_infos = infos

            while env.agents:
                actions = {}

                pa, _ = prisoner_model.predict(np.array(obs["prisoner"]), deterministic=True)
                actions["prisoner"] = int(pa)

                for i in range(args.num_guards):
                    gid = f"guard_{i}"
                    ga, _ = guard_model.predict(np.array(obs[gid]), deterministic=True)
                    actions[gid] = int(ga)

                obs, rewards, terms, truncs, infos = env.step(actions)
                last_infos = infos
                steps += 1

                # rewards dict play modunda hem prisoner hem guards içerir
                ep_ret_p += float(rewards.get("prisoner", 0.0))
                # guards toplamını alıp ortalama yazalım
                gsum = 0.0
                for i in range(args.num_guards):
                    gsum += float(rewards.get(f"guard_{i}", 0.0))
                ep_ret_g += gsum / max(1, args.num_guards)

            outcome = (last_infos.get("prisoner", {}) or {}).get("episode_outcome", "unknown")
            env.close()

            w.writerow([ep, ep_ret_p, ep_ret_g, outcome, steps])

            if ep % 1000 == 0:
                print(f"[{ep}/{args.episodes}] last_outcome={outcome}, last_len={steps}")

    print("✅ Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
