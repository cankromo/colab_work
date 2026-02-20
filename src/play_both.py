import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
import os
from .custom_environment import CustomEnvironment

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONTENT_DIR = os.path.dirname(BASE_DIR)


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    if p.startswith("colab_work/"):
        return os.path.join(CONTENT_DIR, p)
    return os.path.join(BASE_DIR, p)


def save_mp4(frames, path, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # MP4 için en stabil writer
    with imageio.get_writer(path, fps=fps, codec="libx264") as w:
        for f in frames:
            w.append_data(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, default=100)
    p.add_argument("--num_guards", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100)

    p.add_argument("--guard_model_path", type=str, default="colab_work/results/exp5/models/guard_model.zip")
    p.add_argument("--prisoner_model_path", type=str, default="colab_work/results/exp5/models/prisoner_model.zip")
    p.add_argument("--video_path", type=str, default="colab_work/results/exp5/videos/play_both.mp4")

    args = p.parse_args()

    guard_model_path = resolve_path(args.guard_model_path)
    prisoner_model_path = resolve_path(args.prisoner_model_path)
    video_path = resolve_path(args.video_path)

    guard_model = PPO.load(guard_model_path)
    prisoner_model = PPO.load(prisoner_model_path)

    env = CustomEnvironment(
        render_mode="rgb_array",
        grid_size=args.grid_size,
        num_guards=args.num_guards,
        max_steps=args.max_steps,
        training_side="play",
    )

    obs, infos = env.reset()
    frames = []

    while env.agents:
        actions = {}

        # prisoner
        pa, _ = prisoner_model.predict(np.array(obs["prisoner"]), deterministic=True)
        actions["prisoner"] = np.array(pa, dtype=np.float32)

        # guards (same guard model, separate obs)
        for i in range(args.num_guards):
            gid = f"guard_{i}"
            ga, _ = guard_model.predict(np.array(obs[gid]), deterministic=True)
            actions[gid] = np.array(ga, dtype=np.float32)

        obs, rewards, terms, truncs, infos = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    outcome = (infos.get("prisoner", {}) or {}).get("episode_outcome", "unknown")
    fps = env.metadata["render_fps"]
    env.close()

    save_mp4(frames, video_path, fps=fps)
    print("✅ Outcome:", outcome)
    print("✅ Saved video:", video_path)




if __name__ == "__main__":
    main()
