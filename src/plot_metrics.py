import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def _maybe_read_csv(path: str):
    if not os.path.exists(path):
        print("Missing:", path)
        return None
    return pd.read_csv(path)


def plot_train_rewards(log_dir: str, run_name: str, out_path: str) -> None:
    df = _maybe_read_csv(os.path.join(log_dir, run_name, "episode_returns.csv"))
    if df is None or "timesteps" not in df.columns or "ep_return" not in df.columns:
        return

    fig = plt.figure()
    plt.plot(df["timesteps"], df["ep_return"])
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return - {run_name}")
    _save(fig, out_path)


def plot_train_rewards_smoothed(log_dir: str, run_name: str, out_path: str, window: int = 200) -> None:
    df = _maybe_read_csv(os.path.join(log_dir, run_name, "episode_returns.csv"))
    if df is None or "timesteps" not in df.columns or "ep_return" not in df.columns:
        return

    rm = df["ep_return"].rolling(window=window, min_periods=1).mean()
    fig = plt.figure()
    plt.plot(df["timesteps"], rm, label=f"ep_return (roll{window})")
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.title(f"Train Episode Return (Rolling Mean) - {run_name}")
    plt.legend()
    _save(fig, out_path)


def plot_train_loss(log_dir: str, run_name: str, out_path: str) -> None:
    df = _maybe_read_csv(os.path.join(log_dir, run_name, "progress.csv"))
    if df is None:
        return
    if "time/total_timesteps" not in df.columns or "train/loss" not in df.columns:
        print("Missing columns for train loss:", run_name)
        return

    fig = plt.figure()
    plt.plot(df["time/total_timesteps"], df["train/loss"])
    plt.xlabel("timesteps")
    plt.ylabel("loss")
    plt.title(f"Train Loss - {run_name}")
    _save(fig, out_path)


def plot_ppo_diagnostics(log_dir: str, run_name: str, out_path: str) -> None:
    df = _maybe_read_csv(os.path.join(log_dir, run_name, "progress.csv"))
    if df is None or "time/total_timesteps" not in df.columns:
        return

    x = df["time/total_timesteps"]
    series = [
        ("train/approx_kl", "approx_kl"),
        ("train/clip_fraction", "clip_fraction"),
        ("train/entropy_loss", "entropy_loss"),
        ("train/value_loss", "value_loss"),
        ("train/explained_variance", "explained_variance"),
        ("train/policy_gradient_loss", "policy_grad_loss"),
    ]

    fig = plt.figure(figsize=(12, 8))
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
        plt.close(fig)
        print("No PPO diagnostics for:", run_name)
        return

    plt.suptitle(f"PPO Diagnostics - {run_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


def plot_eval_rewards(eval_csv: str, out_path: str, window: int = 200) -> None:
    df = _maybe_read_csv(eval_csv)
    if df is None:
        return
    needed = {"episode", "return_prisoner", "return_guards_mean"}
    if not needed.issubset(df.columns):
        print("Missing columns for eval rewards:", eval_csv)
        return

    rp = df["return_prisoner"].rolling(window=window, min_periods=1).mean()
    rg = df["return_guards_mean"].rolling(window=window, min_periods=1).mean()

    fig = plt.figure()
    plt.plot(df["episode"], rp, label=f"prisoner (roll{window})")
    plt.plot(df["episode"], rg, label=f"guards (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("Evaluation Returns (Rolling Mean)")
    plt.legend()
    _save(fig, out_path)


def plot_eval_winrate(eval_csv: str, out_path: str, window: int = 200) -> None:
    df = _maybe_read_csv(eval_csv)
    if df is None or "episode" not in df.columns or "outcome" not in df.columns:
        return

    prisoner_win = (df["outcome"] == "escaped").astype(float)
    guards_win = (df["outcome"] == "captured").astype(float)
    pw = prisoner_win.rolling(window=window, min_periods=1).mean()
    gw = guards_win.rolling(window=window, min_periods=1).mean()

    fig = plt.figure()
    plt.plot(df["episode"], pw, label=f"prisoner win rate (roll{window})")
    plt.plot(df["episode"], gw, label=f"guards win rate (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("win rate")
    plt.ylim(0.0, 1.0)
    plt.title("Evaluation Win Rate (Rolling Mean)")
    plt.legend()
    _save(fig, out_path)


def plot_eval_outcome_counts(eval_csv: str, out_path: str) -> None:
    df = _maybe_read_csv(eval_csv)
    if df is None or "outcome" not in df.columns:
        return

    counts = df["outcome"].value_counts()
    fig = plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("outcome")
    plt.ylabel("count")
    plt.title("Evaluation Outcome Counts")
    _save(fig, out_path)


def plot_eval_episode_length(eval_csv: str, out_path: str, window: int = 200) -> None:
    df = _maybe_read_csv(eval_csv)
    if df is None or "episode" not in df.columns or "length" not in df.columns:
        return

    length_rm = df["length"].rolling(window=window, min_periods=1).mean()
    fig = plt.figure()
    plt.plot(df["episode"], length_rm, label=f"episode length (roll{window})")
    plt.xlabel("episode")
    plt.ylabel("steps")
    plt.title("Evaluation Episode Length (Rolling Mean)")
    plt.legend()
    _save(fig, out_path)


def plot_train_eval_link(log_dir: str, run_name: str, eval_csv: str, out_path: str, window: int = 200) -> None:
    train_df = _maybe_read_csv(os.path.join(log_dir, run_name, "episode_returns.csv"))
    eval_df = _maybe_read_csv(eval_csv)
    if train_df is None or eval_df is None:
        return
    needed = {"timesteps", "ep_return"}
    if not needed.issubset(train_df.columns) or "outcome" not in eval_df.columns:
        return

    train_rm = train_df["ep_return"].rolling(window=window, min_periods=1).mean()
    prisoner_win = (eval_df["outcome"] == "escaped").mean()
    guards_win = (eval_df["outcome"] == "captured").mean()
    last_t = float(train_df["timesteps"].iloc[-1])

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(train_df["timesteps"], train_rm, label=f"train ep_return (roll{window})")
    ax.scatter([last_t], [train_rm.iloc[-1]], marker="o", color="black", label="last train")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("train episode return")

    ax2 = ax.twinx()
    ax2.scatter([last_t], [prisoner_win], marker="^", color="green", label="eval prisoner win")
    ax2.scatter([last_t], [guards_win], marker="s", color="red", label="eval guards win")
    ax2.set_ylabel("eval win rate")
    ax2.set_ylim(0.0, 1.0)

    plt.title(f"Train vs Eval Link - {run_name}")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    _save(fig, out_path)


def make_all_plots(log_dir: str, eval_csv: str, out_dir: str, run_guards: str = "guards", run_prisoner: str = "prisoner") -> None:
    os.makedirs(out_dir, exist_ok=True)

    plot_train_rewards(log_dir, run_guards, os.path.join(out_dir, "train_reward_guards.png"))
    plot_train_rewards(log_dir, run_prisoner, os.path.join(out_dir, "train_reward_prisoner.png"))

    plot_train_rewards_smoothed(log_dir, run_guards, os.path.join(out_dir, "train_reward_guards_smoothed.png"))
    plot_train_rewards_smoothed(log_dir, run_prisoner, os.path.join(out_dir, "train_reward_prisoner_smoothed.png"))

    plot_train_loss(log_dir, run_guards, os.path.join(out_dir, "train_loss_guards.png"))
    plot_train_loss(log_dir, run_prisoner, os.path.join(out_dir, "train_loss_prisoner.png"))

    plot_ppo_diagnostics(log_dir, run_guards, os.path.join(out_dir, "ppo_diagnostics_guards.png"))
    plot_ppo_diagnostics(log_dir, run_prisoner, os.path.join(out_dir, "ppo_diagnostics_prisoner.png"))

    plot_eval_rewards(eval_csv, os.path.join(out_dir, "eval_returns.png"))
    plot_eval_winrate(eval_csv, os.path.join(out_dir, "eval_winrate.png"))
    plot_eval_outcome_counts(eval_csv, os.path.join(out_dir, "eval_outcomes.png"))
    plot_eval_episode_length(eval_csv, os.path.join(out_dir, "eval_episode_length.png"))

    plot_train_eval_link(log_dir, run_guards, eval_csv, os.path.join(out_dir, "train_eval_link_guards.png"))
    plot_train_eval_link(log_dir, run_prisoner, eval_csv, os.path.join(out_dir, "train_eval_link_prisoner.png"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default=None, help="Example: exp7")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--log_dir", type=str, default=None, help="Override, e.g. results/exp7/log")
    p.add_argument("--eval_csv", type=str, default=None, help="Override, e.g. results/exp7/eval/rollout_returns.csv")
    p.add_argument("--out_dir", type=str, default=None, help="Override, e.g. results/exp7/plots")
    p.add_argument("--run_guards", type=str, default="guards")
    p.add_argument("--run_prisoner", type=str, default="prisoner")
    args = p.parse_args()

    if args.exp_name:
        exp_root = os.path.join(args.results_dir, args.exp_name)
        log_dir = args.log_dir or os.path.join(exp_root, "log")
        eval_csv = args.eval_csv or os.path.join(exp_root, "eval", "rollout_returns.csv")
        out_dir = args.out_dir or os.path.join(exp_root, "plots")
    else:
        log_dir = args.log_dir or "results/exp3/log"
        eval_csv = args.eval_csv or "eval/rollout_returns.csv"
        out_dir = args.out_dir or "plots"

    make_all_plots(
        log_dir=log_dir,
        eval_csv=eval_csv,
        out_dir=out_dir,
        run_guards=args.run_guards,
        run_prisoner=args.run_prisoner,
    )


if __name__ == "__main__":
    main()
