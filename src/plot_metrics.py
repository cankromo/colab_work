import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = "colab_work"   # �� önemli


def plot_train_rewards(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "episode_returns.csv")

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.plot(df["timesteps"], df["ep_return"])
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return - {run_name}")

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_train_loss(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "progress.csv")

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    cols = [c for c in df.columns if c.startswith("train/")]
    if not cols:
        print("No train/* columns found in", path)
        return

    plt.figure()
    for c in cols:
        plt.plot(df["time/total_timesteps"], df[c], label=c)

    plt.xlabel("timesteps")
    plt.ylabel("loss")
    plt.title(f"Train Loss - {run_name}")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_rewards(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.plot(df["episode"], df["return_prisoner"], label="prisoner")
    plt.plot


import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = "colab_work"   # �� önemli


def plot_train_rewards(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "episode_returns.csv")

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.plot(df["timesteps"], df["ep_return"])
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return - {run_name}")

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_train_loss(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "progress.csv")

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    cols = [c for c in df.columns if c.startswith("train/")]
    if not cols:
        print("No train/* columns found in", path)
        return

    plt.figure()
    for c in cols:
        plt.plot(df["time/total_timesteps"], df[c], label=c)

    plt.xlabel("timesteps")
    plt.ylabel("loss")
    plt.title(f"Train Loss - {run_name}")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_rewards(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("❌ Not found:", path)
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.plot(df["episode"], df["return_prisoner"], label="prisoner")
    plt.plot(df["episode"], df["return_guards_mean"], label="guards")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("Evaluation 10k Returns")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str, default="results/exp3/log")
    p.add_argument("--run_guards", type=str, default="guards")
    p.add_argument("--run_prisoner", type=str, default="prisoner")
    p.add_argument("--eval_csv", type=str, default="eval/rollout_returns.csv")
    args = p.parse_args()

    plot_train_rewards(args.log_dir, args.run_guards, "plots/train_reward_guards.png")
    plot_train_rewards(args.log_dir, args.run_prisoner, "plots/train_reward_prisoner.png")

    plot_train_loss(args.log_dir, args.run_guards, "plots/train_loss_guards.png")
    plot_train_loss(args.log_dir, args.run_prisoner, "plots/train_loss_prisoner.png")

    plot_eval_rewards(args.eval_csv, "plots/eval_10k_returns.png")


if __name__ == "__main__":
    main()