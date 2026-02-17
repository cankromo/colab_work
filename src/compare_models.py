import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import numpy as np
from stable_baselines3 import PPO

from .custom_environment import CustomEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--reward_mode", type=str, default="legacy", choices=["legacy", "dynamic"])

    p.add_argument("--guard_model_path", type=str, default="models/guard_model.zip")
    p.add_argument("--prisoner_model_path", type=str, default="models/prisoner_model.zip")

    # eğer prisoner modelin yoksa heuristic ile test etmek için:
    p.add_argument("--prisoner_heuristic", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)

    guard_model = PPO.load(args.guard_model_path)
    prisoner_model = None if args.prisoner_heuristic else PPO.load(args.prisoner_model_path)

    results = {"captured": 0, "escaped": 0, "timeout": 0, "unknown": 0}
    total_len = 0

    for ep in range(1, args.episodes + 1):
        env = CustomEnvironment(
            render_mode=None,
            grid_size=args.grid_size,
            num_guards=args.num_guards,
            max_steps=args.max_steps,
            training_side="play",
            reward_mode=args.reward_mode,
            seed=args.seed + ep,  # her episode farklı olsun
        )

        obs, infos = env.reset()
        steps = 0
        last_infos = infos

        while env.agents:
            actions = {}

            # prisoner action
            if "prisoner" in env.agents:
                if args.prisoner_heuristic:
                    # env içindeki heuristic'i kullanmak için:
                    # play modunda dışarıdan action beklediği için burada heuristik hesaplıyoruz
                    # (env'deki heuristik aynı mantık)
                    prisoner_pos = env.agents_obj["prisoner"]["pos"]
                    escape_pos = env.escape_pos
                    guards = [env.agents_obj[f"guard_{i}"]["pos"] for i in range(args.num_guards)]
                    if guards:
                        dists = [np.linalg.norm(prisoner_pos - g) for g in guards]
                        closest = guards[int(np.argmin(dists))]
                        escape_vec = escape_pos - prisoner_pos
                        avoid_vec = prisoner_pos - closest
                        final_vec = 2 * escape_vec + avoid_vec
                        if abs(final_vec[0]) > abs(final_vec[1]):
                            actions["prisoner"] = 1 if final_vec[0] > 0 else 0
                        else:
                            actions["prisoner"] = 3 if final_vec[1] > 0 else 2
                    else:
                        actions["prisoner"] = np.random.randint(0, 4)
                else:
                    pa, _ = prisoner_model.predict(np.array(obs["prisoner"]), deterministic=True)
                    actions["prisoner"] = int(pa)

            # guards actions
            for i in range(args.num_guards):
                gid = f"guard_{i}"
                ga, _ = guard_model.predict(np.array(obs[gid]), deterministic=True)
                actions[gid] = int(ga)

            obs, rewards, terms, truncs, infos = env.step(actions)
            last_infos = infos
            steps += 1

        env.close()

        # outcome
        outcome = (last_infos.get("prisoner", {}) or {}).get("episode_outcome", "unknown")
        if outcome not in results:
            outcome = "unknown"

        results[outcome] += 1
        total_len += steps

        # ara log
        if ep % 100 == 0:
            done = ep
            print(
                f"[{done}/{args.episodes}] "
                f"capture={results['captured']/done:.3f}, "
                f"escape={results['escaped']/done:.3f}, "
                f"timeout={results['timeout']/done:.3f}, "
                f"avg_len={total_len/done:.2f}"
            )

    n = args.episodes
    print("\n=== Final Stats ===")
    print(f"Episodes: {n}")
    print(f"Captured: {results['captured']} ({results['captured']/n:.3%})")
    print(f"Escaped : {results['escaped']} ({results['escaped']/n:.3%})")
    print(f"Timeout : {results['timeout']} ({results['timeout']/n:.3%})")
    print(f"Unknown : {results['unknown']} ({results['unknown']/n:.3%})")
    print(f"Avg ep length: {total_len/n:.2f}")


if __name__ == "__main__":
    main()
