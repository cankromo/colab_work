from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class OutcomeStatsCallback(BaseCallback):
    """
    capture/escape/timeout istatistiklerini eğitim sırasında yazdırır.
    Outcome'u env'in info'sundaki 'episode_outcome' alanından okur.
    """

    def __init__(self, print_every: int = 5000, window: int = 200, verbose=0):
        super().__init__(verbose)
        self.print_every = int(print_every)
        self.window = int(window)
        self.outcomes = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            outcome = info.get("episode_outcome", None)

            # Fallback: info yoksa time limit vs.
            if outcome not in ("captured", "escaped", "timeout"):
                # SB3 bazı durumlarda TimeLimit.truncated bırakır
                if info.get("TimeLimit.truncated", False):
                    outcome = "timeout"
                else:
                    outcome = "timeout"

            self.outcomes.append(outcome)

        if self.n_calls % self.print_every == 0 and len(self.outcomes) >= max(10, self.window // 2):
            total = len(self.outcomes)
            captures = sum(1 for x in self.outcomes if x == "captured")
            escapes = sum(1 for x in self.outcomes if x == "escaped")
            timeouts = sum(1 for x in self.outcomes if x == "timeout")

            print(f"[t={self.num_timesteps}] capture={100*captures/total:.2f}%, "
                  f"escape={100*escapes/total:.2f}%, timeout={100*timeouts/total:.2f}%, "
                  f"ep_cnt={total}")

        return True
