import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_train_rewards(log_dir, run_name, out_path):
    path = os.path.join(log_dir, run_name, "episode_returns.csv")
    df = pd.read_csv(path)
    plt.figure()
    plt.plot(df["timesteps"], df["ep_return"])
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return - {run_name}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_train_loss(log_dir, run_name, out_path):
    # SB3 CSV logger default: progress.csv
    path = os.path.join(log_dir, run_name, "progress.csv")
    df = pd.read_csv(path)

    # En temel losslar:
    cols = [c for c in ["train/loss", "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss"] if c in df.columns]
    if not cols:
        raise RuntimeError(f"No train/* loss columns found in {path}. Columns are: {list(df.columns)[:30]} ...")

    plt.figure()
    for c in cols:
        plt.plot(df["time/total_timesteps"], df[c], label=c)
    plt.xlabel("timesteps")
    plt.ylabel("loss")
    plt.title(f"Train Loss Curves - {run_name}")
    plt.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eval_rewards(eval_csv, out_path):
    df = pd.read_csv(eval_csv)
    plt.figure()
    plt.plot(df["episode"], df["return_prisoner"], label="prisoner return")
    plt.plot(df["episode"], df["return_guards_mean"], label="guards mean return")
    plt.xlabel("episode")
    plt.ylabel("episode return")
    plt.title("Evaluation (10k) Episode Returns")
    plt.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # paths
    log_dir = "logs"
    eval_csv = "eval/rollout_returns.csv"

    plot_train_rewards(log_dir, "guards", "plots/train_reward_guards.png")
    plot_train_rewards(log_dir, "prisoner", "plots/train_reward_prisoner.png")

    plot_train_loss(log_dir, "guards", "plots/train_loss_guards.png")
    plot_train_loss(log_dir, "prisoner", "plots/train_loss_prisoner.png")

    plot_eval_rewards(eval_csv, "plots/eval_10k_returns.png")

    print("✅ Plots saved under plots/")


if __name__ == "__main__":
    main()