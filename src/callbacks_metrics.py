import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeReturnLogger(BaseCallback):
    """
    VecMonitor/Monitor ile gelen info['episode'] alanından episodic return'u alır ve
    ayrı bir CSV'ye yazar: logs/<run_name>/episode_returns.csv

    Not: VecMonitor sarmalarsan info içine 'episode' gelir.
    """
    def __init__(self, log_dir: str, run_name: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_name = run_name
        self.path = os.path.join(log_dir, run_name, "episode_returns.csv")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = None

    def _on_training_start(self):
        mode = "a" if os.path.exists(self.path) and os.path.getsize(self.path) > 0 else "w"
        self._fh = open(self.path, mode, encoding="utf-8")
        if mode == "w":
            self._fh.write("timesteps,ep_return,ep_length\n")
        self._fh.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            # ep: {"r": return, "l": length, "t": time}
            self._fh.write(f"{self.num_timesteps},{ep['r']},{ep['l']}\n")
        self._fh.flush()
        return True

    def _on_training_end(self):
        if self._fh:
            self._fh.close()
            self._fh = None
