import argparse
import os
import time

import imageio
import numpy as np
from stable_baselines3 import PPO

from IPython.display import display, clear_output
from PIL import Image
from custom_environment import CustomEnvironment

os.environ["SDL_VIDEODRIVER"] = "dummy"


def evaluate(args):
    env = CustomEnvironment(
        render_mode="rgb_array",
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        max_steps=args.max_steps,
        seed=args.seed
    )
    model = PPO.load(args.model_path)
    obs, infos = env.reset()

    frames = [] if not args.inline else None

    try:
        while env.agents:
            actions = {}
            for agent_id in env.agents:
                a, _ = model.predict(np.array(obs[agent_id]), deterministic=True)
                actions[agent_id] = int(a)

            obs, rewards, terminations, truncations, infos = env.step(actions)

            frame = env.render()
            if frame is not None:
                if args.inline:
                    clear_output(wait=True)
                    display(Image.fromarray(frame))
                    time.sleep(1 / env.metadata.get("render_fps", 10))
                else:
                    frames.append(frame)

            if not env.agents:
                if args.inline:
                    print("Oyun bitti. Outcome:", (infos.get("guard_0", {}) or {}).get("episode_outcome", "unknown"))
                break

    except KeyboardInterrupt:
        if args.inline:
            print("\nDeğerlendirme kullanıcı tarafından durduruldu.")
    finally:
        env.close()
        if args.inline:
            print("Ortam kapatıldı.")

    if frames is not None and frames:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        with imageio.get_writer("videos/gameplay.mp4" , fps=4,codec ="libx264") as w:
          for f in frames:
              w.append_data(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, default="videos/gameplay.mp4")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--num_guards", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inline", action="store_true")
    args = parser.parse_args()
    evaluate(args)