import argparse
import os

import imageio
import numpy as np
from stable_baselines3 import PPO

from custom_environment import CustomEnvironment

os.environ["SDL_VIDEODRIVER"] = "dummy"


def evaluate(args):
    """Evaluate a trained model and save a video."""
    model = PPO.load(args.model_path)

    env = CustomEnvironment(render_mode="rgb_array", grid_size=args.grid_size, num_guards=args.num_guards)
    obs, infos = env.reset()

    frames = []

    while env.agents:
        actions = {}
        for agent_id in env.agents:  # only guards
            a, _ = model.predict(np.array(obs[agent_id]), deterministic=True)
            actions[agent_id] = int(a)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    if frames:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        imageio.mimsave(args.video_path, frames, fps=env.metadata["render_fps"])
        print("Saved video to:", args.video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.zip file).")
    parser.add_argument("--video_path", type=str, default="videos/gameplay.mp4", help="Path to save the gameplay video.")
    parser.add_argument("--grid_size", type=int, default=25, help="Size of the grid for evaluation.")
    parser.add_argument("--num_guards", type=int, default=2, help="Number of guards for evaluation.")
    args = parser.parse_args()
    evaluate(args)

