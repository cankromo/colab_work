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
    """Evaluate a trained model and save a video."""
    # Ortamı 'rgb_array' modunda başlatıyoruz, çünkü her kareyi bir resim olarak alacağız.
    env = CustomEnvironment(render_mode="rgb_array", grid_size=args.grid_size, num_guards=args.num_guards)
    model = PPO.load(args.model_path)
    obs, infos = env.reset()

    # --inline bayrağı kullanılmadıysa kareleri biriktirmek için bir liste oluştur.
    frames = [] if not args.inline else None

    try:
        while env.agents:
            actions = {}
            for agent_id in env.agents:  # Sadece gardiyanlar için
                a, _ = model.predict(np.array(obs[agent_id]), deterministic=True)
                actions[agent_id] = int(a)

            obs, rewards, terminations, truncations, infos = env.step(actions)

            frame = env.render()
            if frame is not None:
                if args.inline:
                    # Canlı mod: Her kareyi hücrede göster
                    clear_output(wait=True)
                    img = Image.fromarray(frame)
                    display(img)
                    # Oyun hızını yavaşlat
                    time.sleep(1 / env.metadata.get("render_fps", 10))
                else:
                    # Video modu: Kareyi listeye ekle
                    frames.append(frame)

            if not env.agents:
                if args.inline:
                    print("Oyun bitti.")
                break
    except KeyboardInterrupt:
        if args.inline:
            print("\nDeğerlendirme kullanıcı tarafından durduruldu.")
    finally:
        env.close()
        if args.inline:
            print("Ortam kapatıldı.")

    # Eğer 'frames' listesi doluysa (yani --inline modu kapalıysa) videoyu kaydet
    if frames is not None and frames:
        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        imageio.mimsave(args.video_path, frames, fps=env.metadata["render_fps"])
        print("Saved video to:", args.video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent and record a video or display it inline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.zip file).")
    parser.add_argument("--video_path", type=str, default="videos/gameplay.mp4", help="Path to save the gameplay video.")
    parser.add_argument("--grid_size", type=int, default=25, help="Size of the grid for evaluation.")
    parser.add_argument("--num_guards", type=int, default=2, help="Number of guards for evaluation.")
    parser.add_argument("--inline", action="store_true", help="Display gameplay inline in a notebook instead of saving a video.")
    args = parser.parse_args()
    evaluate(args)
