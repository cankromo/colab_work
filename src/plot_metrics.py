import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.getenv("PLOT_BASE_DIR", "colab_work")


def plot_train_rewards(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "episode_returns.csv")

    if not os.path.exists(path):
        print("Not found:", path)
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
        print("Not found:", path)
        return

    df = pd.read_csv(path)

    if "train/loss" not in df.columns:
        print("No train/loss column found in", path)
        return

    plt.figure()
    plt.plot(df["time/total_timesteps"], df["train/loss"])

    plt.xlabel("timesteps")
    plt.ylabel("loss")
    plt.title(f"Train Loss - {run_name}")
    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_rewards(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)

    plt.figure()
    window = 200
    rp = df["return_prisoner"].rolling(window=window, min_periods=1).mean()
    rg = df["return_guards_mean"].rolling(window=window, min_periods=1).mean()
    plt.plot(df["episode"], rp, label=f"prisoner (roll{window})")
    plt.plot(df["episode"], rg, label=f"guards (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("Evaluation 10k Returns (Rolling Mean)")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_winrate(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)

    if "outcome" not in df.columns:
        print("No outcome column found in", path)
        return

    prisoner_win = (df["outcome"] == "escaped").astype(float)
    guards_win = (df["outcome"] == "captured").astype(float)

    window = 200
    pw = prisoner_win.rolling(window=window, min_periods=1).mean()
    gw = guards_win.rolling(window=window, min_periods=1).mean()

    plt.figure()
    plt.plot(df["episode"], pw, label=f"prisoner win rate (roll{window})")
    plt.plot(df["episode"], gw, label=f"guards win rate (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("win rate")
    plt.ylim(0.0, 1.0)
    plt.title("Evaluation 10k Win Rate (Rolling Mean)")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_outcome_counts(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        print("No outcome column found in", path)
        return

    counts = df["outcome"].value_counts()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("outcome")
    plt.ylabel("count")
    plt.title("Evaluation Outcome Counts")

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_eval_episode_length(eval_csv, out_path):
    path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)
    if "length" not in df.columns:
        print("No length column found in", path)
        return

    window = 200
    length_rm = df["length"].rolling(window=window, min_periods=1).mean()

    plt.figure()
    plt.plot(df["episode"], length_rm, label=f"episode length (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("steps")
    plt.title("Evaluation Episode Length (Rolling Mean)")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_train_rewards_smoothed(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "episode_returns.csv")

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)
    window = 200
    rm = df["ep_return"].rolling(window=window, min_periods=1).mean()

    plt.figure()
    plt.plot(df["timesteps"], rm, label=f"ep_return (roll{window})")
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return (Rolling Mean) - {run_name}")
    plt.legend()

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_ppo_diagnostics(log_dir, run_name, out_path):
    path = os.path.join(BASE_DIR, log_dir, run_name, "progress.csv")

    if not os.path.exists(path):
        print("Not found:", path)
        return

    df = pd.read_csv(path)
    x = df.get("time/total_timesteps")
    if x is None:
        print("No time/total_timesteps column found in", path)
        return

    series = [
        ("train/approx_kl", "approx_kl"),
        ("train/clip_fraction", "clip_fraction"),
        ("train/entropy_loss", "entropy_loss"),
        ("train/value_loss", "value_loss"),
        ("train/explained_variance", "explained_variance"),
        ("train/policy_gradient_loss", "policy_grad_loss"),
    ]

    plt.figure(figsize=(12, 8))
    rows, cols = 2, 3
    plot_idx = 1
    for col, title in series:
        if col not in df.columns:
            continue
        ax = plt.subplot(rows, cols, plot_idx)
        ax.plot(x, df[col])
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        plot_idx += 1

    if plot_idx == 1:
        print("No PPO diagnostic columns found in", path)
        return

    plt.suptitle(f"PPO Diagnostics - {run_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    full_out = os.path.join(BASE_DIR, out_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    plt.savefig(full_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", full_out)


def plot_train_eval_link(log_dir, run_name, eval_csv, out_path):
    train_path = os.path.join(BASE_DIR, log_dir, run_name, "episode_returns.csv")
    eval_path = os.path.join(BASE_DIR, eval_csv)

    if not os.path.exists(train_path):
        print("Not found:", train_path)
        return
    if not os.path.exists(eval_path):
        print("Not found:", eval_path)
        return

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    window = 200
    train_rm = train_df["ep_return"].rolling(window=window, min_periods=1).mean()

    prisoner_win = (eval_df.get("outcome") == "escaped").mean()
    guards_win = (eval_df.get("outcome") == "captured").mean()

    last_t = float(train_df["timesteps"].iloc[-1])

    plt.figure()
    plt.plot(train_df["timesteps"], train_rm, label=f"train ep_return (roll{window})")
    plt.scatter([last_t], [train_rm.iloc[-1]], marker="o", color="black", label="last train")

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.scatter([last_t], [prisoner_win], marker="^", color="green", label="eval prisoner win")
    ax2.scatter([last_t], [guards_win], marker="s", color="red", label="eval guards win")
    ax2.set_ylabel("eval win rate")
    ax2.set_ylim(0.0, 1.0)

    ax.set_xlabel("timesteps")
    ax.set_ylabel("train episode return")
    plt.title(f"Train vs Eval Link - {run_name}")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

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
    p.add_argument("--plot_dir", type=str, default="plots")
    args = p.parse_args()

    plot_train_rewards(args.log_dir, args.run_guards, os.path.join(args.plot_dir, "train_reward_guards.png"))
    plot_train_rewards(args.log_dir, args.run_prisoner, os.path.join(args.plot_dir, "train_reward_prisoner.png"))

    plot_train_loss(args.log_dir, args.run_guards, os.path.join(args.plot_dir, "train_loss_guards.png"))
    plot_train_loss(args.log_dir, args.run_prisoner, os.path.join(args.plot_dir, "train_loss_prisoner.png"))

    plot_eval_rewards(args.eval_csv, os.path.join(args.plot_dir, "eval_10k_returns.png"))
    plot_eval_winrate(args.eval_csv, os.path.join(args.plot_dir, "eval_10k_winrate.png"))
    plot_eval_outcome_counts(args.eval_csv, os.path.join(args.plot_dir, "eval_10k_outcomes.png"))
    plot_eval_episode_length(args.eval_csv, os.path.join(args.plot_dir, "eval_10k_episode_length.png"))

    plot_train_rewards_smoothed(args.log_dir, args.run_guards, os.path.join(args.plot_dir, "train_reward_guards_smoothed.png"))
    plot_train_rewards_smoothed(args.log_dir, args.run_prisoner, os.path.join(args.plot_dir, "train_reward_prisoner_smoothed.png"))

    plot_ppo_diagnostics(args.log_dir, args.run_guards, os.path.join(args.plot_dir, "ppo_diagnostics_guards.png"))
    plot_ppo_diagnostics(args.log_dir, args.run_prisoner, os.path.join(args.plot_dir, "ppo_diagnostics_prisoner.png"))

    plot_train_eval_link(args.log_dir, args.run_guards, args.eval_csv, os.path.join(args.plot_dir, "train_eval_link_guards.png"))
    plot_train_eval_link(args.log_dir, args.run_prisoner, args.eval_csv, os.path.join(args.plot_dir, "train_eval_link_prisoner.png"))


if __name__ == "__main__":
    main()
