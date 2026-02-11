from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class OutcomeStatsCallback(BaseCallback):
    """
    A custom callback to print capture/escape/truncation statistics during training.
    """

    def __init__(self, print_every: int, window: int, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self.window = window
        # A deque to store the outcomes of the last `window` episodes
        self.outcomes = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # `dones` is a boolean array indicating if an episode has ended in any of the parallel environments.
        # `infos` is a list of dictionaries, one for each environment.
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # An episode has just finished. The info dict contains the episode stats.
                info = self.locals["infos"][i]
                episode_reward = info.get("episode", {}).get("r", 0)

                # Determine the outcome based on the terminal reward.
                # +1 for capture, -1 for escape.
                if episode_reward > 0.5:
                    self.outcomes.append("capture")
                elif episode_reward < -0.5:
                    self.outcomes.append("escape")
                else:
                    self.outcomes.append("truncated")

        # Print stats every `print_every` steps if the deque is full.
        if self.n_calls % self.print_every == 0 and len(self.outcomes) == self.window:
            total = len(self.outcomes)
            captures = self.outcomes.count("capture")
            escapes = self.outcomes.count("escape")
            truncations = self.outcomes.count("truncated")

            print(f"--- Timestep {self.n_calls} ---")
            print(f"Stats over last {self.window} episodes:")
            print(f"  Capture rate: {100 * captures / total:.2f}%")
            print(f"  Escape rate:  {100 * escapes / total:.2f}%")
            print(f"  Timeout rate: {100 * truncations / total:.2f}%")
            print("--------------------")

        return True
