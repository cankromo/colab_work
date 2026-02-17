import functools
import random
from copy import copy

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv


class CustomEnvironment(ParallelEnv):
    """
    training_side:
      - "guards": dışarıdan guard actions gelir, prisoner env içinde (heuristic veya sabit prisoner model)
      - "prisoner": dışarıdan prisoner action gelir, guards env içinde (sabit guard model)
      - "play": dışarıdan hem guards hem prisoner action gelir (play_both.py kullanır)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "custom_environment_v0",
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode=None,
        grid_size=10,
        num_guards=2,
        max_steps=100,
        seed=None,
        training_side="guards",
        # Opponent policies
        guard_model_path=None,
        prisoner_model_path=None,
        prisoner_use_heuristic=True,
        # Reward shaping weights
        reward_mode="legacy",
        guard_escape_penalty_lambda=0.05,
        guard_time_penalty=None,
        prisoner_guard_penalty_lambda=0.05,
        prisoner_time_penalty=None,
    ):
        self.render_mode = render_mode
        self.grid_size = int(grid_size)
        self.num_guards = int(num_guards)
        self.max_steps = int(max_steps)
        self.training_side = training_side

        self.guard_model_path = guard_model_path
        self.prisoner_model_path = prisoner_model_path
        self.prisoner_use_heuristic = prisoner_use_heuristic
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {"legacy", "dynamic"}:
            raise ValueError("reward_mode must be one of: legacy, dynamic")

        self.guard_escape_penalty_lambda = float(guard_escape_penalty_lambda)
        self.prisoner_guard_penalty_lambda = float(prisoner_guard_penalty_lambda)
        # Keep per-episode time penalty bounded across different max_steps.
        # With this scaling, timeout contributes at most ~0.5 penalty per episode.
        if guard_time_penalty is None:
            self.guard_time_penalty = 0.5 / max(1, self.max_steps)
        else:
            self.guard_time_penalty = float(guard_time_penalty)
        if prisoner_time_penalty is None:
            self.prisoner_time_penalty = 0.5 / max(1, self.max_steps)
        else:
            self.prisoner_time_penalty = float(prisoner_time_penalty)

        # dynamic agents list based on training_side
        if self.training_side == "guards":
            self.possible_agents = [f"guard_{i}" for i in range(self.num_guards)]
        elif self.training_side == "prisoner":
            self.possible_agents = ["prisoner"]
        elif self.training_side == "play":
            self.possible_agents = ["prisoner"] + [f"guard_{i}" for i in range(self.num_guards)]
        else:
            raise ValueError("training_side must be one of: guards, prisoner, play")

        self.agents = []

        # world state
        self.agents_obj = {}
        self.escape_pos = np.zeros(2, dtype=np.int32)
        self.timestep = 0

        # shaping baselines
        self._prev_total_guard_dist = None
        self._prev_escape_dist = None
        self._prev_prisoner_escape_dist = None
        self._prev_prisoner_guard_dist = None

        # opponent models (lazy loaded)
        self._guard_model = None
        self._prisoner_model = None

        # render stuff
        self.window = None
        self.clock = None
        self.font = None
        if self.render_mode in ["human", "rgb_array"]:
            self.window_size = 512
            self.cell_size = self.window_size / self.grid_size

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # -------------------------
    # Helpers
    # -------------------------
    def _chebyshev_dist(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.linalg.norm(a - b, ord=np.inf))

    def _apply_move(self, pos: np.ndarray, action: int, move_count: int = 1) -> None:
        for _ in range(move_count):
            if action == 0 and pos[0] > 0:
                pos[0] -= 1
            elif action == 1 and pos[0] < self.grid_size - 1:
                pos[0] += 1
            elif action == 2 and pos[1] > 0:
                pos[1] -= 1
            elif action == 3 and pos[1] < self.grid_size - 1:
                pos[1] += 1

    def _total_guard_distance_l2(self) -> float:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        total = 0.0
        for i in range(self.num_guards):
            gpos = self.agents_obj[f"guard_{i}"]["pos"]
            total += float(np.linalg.norm(prisoner_pos - gpos))
        return total

    def _escape_distance_l2(self) -> float:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        return float(np.linalg.norm(prisoner_pos - self.escape_pos))

    def _relative_progress(self, prev_value: float, curr_value: float) -> float:
        """
        Relative step progress bounded to [-1, 1].
        Positive means current value improved versus previous.
        """
        prev_value = float(prev_value)
        if prev_value <= 1e-8:
            return 0.0
        ratio = (prev_value - float(curr_value)) / prev_value
        return float(np.clip(ratio, -1.0, 1.0))

    # -------------------------
    # Observations (A)
    # -------------------------
    def _norm(self, v: np.ndarray) -> np.ndarray:
        denom = float(max(1, self.grid_size - 1))
        return v.astype(np.float32) / denom

    def _get_guard_obs(self, guard_id: str) -> np.ndarray:
        """A (guards): my_pos + prisoner_pos + other_guards + escape_pos + id_onehot"""
        k = int(guard_id.split("_")[1])

        my_pos = self._norm(self.agents_obj[guard_id]["pos"])
        prisoner_pos = self._norm(self.agents_obj["prisoner"]["pos"])
        escape_pos = self._norm(self.escape_pos)

        others = []
        for i in range(self.num_guards):
            if i == k:
                continue
            others.append(self._norm(self.agents_obj[f"guard_{i}"]["pos"]))
        others = np.concatenate(others) if others else np.array([], dtype=np.float32)

        id_onehot = np.zeros(self.num_guards, dtype=np.float32)
        id_onehot[k] = 1.0

        return np.concatenate([my_pos, prisoner_pos, others, escape_pos, id_onehot]).astype(np.float32)

    def _get_prisoner_obs(self) -> np.ndarray:
        """A (prisoner): prisoner_pos + all_guards_pos + escape_pos"""
        prisoner_pos = self._norm(self.agents_obj["prisoner"]["pos"])
        guards = [self._norm(self.agents_obj[f"guard_{i}"]["pos"]) for i in range(self.num_guards)]
        guards = np.concatenate(guards) if guards else np.array([], dtype=np.float32)
        escape_pos = self._norm(self.escape_pos)
        return np.concatenate([prisoner_pos, guards, escape_pos]).astype(np.float32)

    def _get_observation(self, agent: str) -> np.ndarray:
        if agent == "prisoner":
            return self._get_prisoner_obs()
        return self._get_guard_obs(agent)

    # -------------------------
    # Opponent actions
    # -------------------------
    def _lazy_load_guard_model(self):
        if self._guard_model is None:
            if not self.guard_model_path:
                raise ValueError("guard_model_path is required to control guards with a model")
            from stable_baselines3 import PPO
            self._guard_model = PPO.load(self.guard_model_path)

    def _lazy_load_prisoner_model(self):
        if self._prisoner_model is None:
            if not self.prisoner_model_path:
                raise ValueError("prisoner_model_path is required to control prisoner with a model")
            from stable_baselines3 import PPO
            self._prisoner_model = PPO.load(self.prisoner_model_path)

    def _prisoner_heuristic_action(self) -> int:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        escape_pos = self.escape_pos
        guards = [self.agents_obj[f"guard_{i}"]["pos"] for i in range(self.num_guards)]
        if not guards:
            return random.randint(0, 3)

        distances = [np.linalg.norm(prisoner_pos - g) for g in guards]
        closest_guard = guards[int(np.argmin(distances))]

        escape_vec = escape_pos - prisoner_pos
        avoid_vec = prisoner_pos - closest_guard
        final_vec = 2 * escape_vec + avoid_vec

        if abs(final_vec[0]) > abs(final_vec[1]):
            return 1 if final_vec[0] > 0 else 0
        return 3 if final_vec[1] > 0 else 2

    def _get_prisoner_action(self) -> int:
        if self.training_side == "play":
            raise RuntimeError("play mode expects prisoner action from outside")
        if self.prisoner_use_heuristic:
            return self._prisoner_heuristic_action()
        self._lazy_load_prisoner_model()
        obs = self._get_prisoner_obs()
        a, _ = self._prisoner_model.predict(obs, deterministic=True)
        return int(a)

    def _get_guard_actions_from_model(self) -> dict:
        self._lazy_load_guard_model()
        actions = {}
        for i in range(self.num_guards):
            gid = f"guard_{i}"
            obs = self._get_guard_obs(gid)
            a, _ = self._guard_model.predict(obs, deterministic=True)
            actions[gid] = int(a)
        return actions

    # -------------------------
    # PettingZoo API
    # -------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # sample initial state (avoid immediate terminal)
        for _ in range(200):
            self.agents_obj = {"prisoner": {"role": "prisoner", "pos": np.array([0, 0], dtype=np.int32)}}
            for i in range(self.num_guards):
                self.agents_obj[f"guard_{i}"] = {
                    "role": "guard",
                    "pos": np.array(
                        [random.randint(1, self.grid_size - 1), random.randint(1, self.grid_size - 1)],
                        dtype=np.int32,
                    ),
                }
            self.escape_pos = np.array(
                [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)],
                dtype=np.int32,
            )

            prisoner_pos = self.agents_obj["prisoner"]["pos"]
            too_close_guard = any(
                self._chebyshev_dist(prisoner_pos, self.agents_obj[f"guard_{i}"]["pos"]) <= 1
                for i in range(self.num_guards)
            )
            too_close_escape = (self._chebyshev_dist(prisoner_pos, self.escape_pos) <= 1)

            if not too_close_guard and not too_close_escape:
                break

        # shaping baselines
        self._prev_total_guard_dist = self._total_guard_distance_l2()
        self._prev_escape_dist = self._escape_distance_l2()
        self._prev_prisoner_escape_dist = self._escape_distance_l2()
        self._prev_prisoner_guard_dist = self._total_guard_distance_l2()

        observations = {a: self._get_observation(a) for a in self.agents}
        infos = {a: {"episode_outcome": "running"} for a in self.agents}

        if self.render_mode in ["human", "rgb_array"]:
            self.render()
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.timestep += 1

        # --- build full action set (prisoner + all guards)
        full_actions = {}

        if self.training_side == "guards":
            # guards come from outside, prisoner from env policy
            for i in range(self.num_guards):
                gid = f"guard_{i}"
                full_actions[gid] = int(actions.get(gid, self.action_space(gid).sample()))
            full_actions["prisoner"] = self._get_prisoner_action()

        elif self.training_side == "prisoner":
            # prisoner from outside, guards from fixed guard model
            full_actions["prisoner"] = int(actions.get("prisoner", self.action_space("prisoner").sample()))
            full_actions.update(self._get_guard_actions_from_model())

        elif self.training_side == "play":
            # both provided
            for i in range(self.num_guards):
                gid = f"guard_{i}"
                full_actions[gid] = int(actions.get(gid, self.action_space(gid).sample()))
            full_actions["prisoner"] = int(actions.get("prisoner", self.action_space("prisoner").sample()))

        # --- apply moves
        # prisoner moves twice
        self._apply_move(self.agents_obj["prisoner"]["pos"], full_actions["prisoner"], move_count=2)
        # guards move once
        for i in range(self.num_guards):
            gid = f"guard_{i}"
            self._apply_move(self.agents_obj[gid]["pos"], full_actions[gid], move_count=1)

        prisoner_pos = self.agents_obj["prisoner"]["pos"]

        captured = any(
            self._chebyshev_dist(prisoner_pos, self.agents_obj[f"guard_{i}"]["pos"]) <= 1
            for i in range(self.num_guards)
        )
        escaped = (self._chebyshev_dist(prisoner_pos, self.escape_pos) <= 1)

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        infos = {a: {} for a in self.agents}

        # --- terminal rewards (zero-sum)
        if captured:
            # guards win
            if self.training_side == "prisoner":
                rewards["prisoner"] = -1.0
                terminations["prisoner"] = True
            else:
                for a in rewards:
                    rewards[a] = 1.0
                    terminations[a] = True
            outcome = "captured"

        elif escaped:
            # prisoner wins
            if self.training_side == "prisoner":
                rewards["prisoner"] = 1.0
                terminations["prisoner"] = True
            else:
                for a in rewards:
                    rewards[a] = -1.0
                    terminations[a] = True
            outcome = "escaped"

        else:
            outcome = "running"

            if self.training_side == "guards":
                total_guard_dist = self._total_guard_distance_l2()
                esc_dist = self._escape_distance_l2()

                if self.reward_mode == "dynamic":
                    # Scale by relative per-step progress to reduce grid-size bias.
                    approach_reward = self._relative_progress(self._prev_total_guard_dist, total_guard_dist)
                    escape_step_progress = self._relative_progress(self._prev_escape_dist, esc_dist)
                    escape_penalty = -self.guard_escape_penalty_lambda * escape_step_progress
                else:
                    # Legacy shaping.
                    dist_delta = self._prev_total_guard_dist - total_guard_dist
                    max_ref = float(self.grid_size * self.num_guards * 1.5)
                    approach_reward = float(dist_delta) / max(1.0, max_ref)

                    max_esc = float(np.sqrt(2) * (self.grid_size - 1))
                    escape_progress = (max_esc - esc_dist) / max(1.0, max_esc)
                    escape_progress = float(np.clip(escape_progress, 0.0, 1.0))
                    escape_penalty = -self.guard_escape_penalty_lambda * escape_progress

                self._prev_total_guard_dist = total_guard_dist
                self._prev_escape_dist = esc_dist
                shaped = approach_reward + escape_penalty - self.guard_time_penalty
                for a in rewards:
                    rewards[a] = float(shaped)

            elif self.training_side == "prisoner":
                esc_dist = self._escape_distance_l2()
                gdist = self._total_guard_distance_l2()

                if self.reward_mode == "dynamic":
                    # Positive when prisoner gets closer to escape this step.
                    progress_reward = self._relative_progress(self._prev_prisoner_escape_dist, esc_dist)
                    # Positive when prisoner increases separation from guards.
                    separation_reward = -self._relative_progress(self._prev_prisoner_guard_dist, gdist)
                    guard_term = self.prisoner_guard_penalty_lambda * separation_reward
                    shaped = progress_reward + guard_term - self.prisoner_time_penalty
                else:
                    # Legacy shaping.
                    delta = self._prev_prisoner_escape_dist - esc_dist
                    max_esc = float(np.sqrt(2) * (self.grid_size - 1))
                    progress_reward = float(delta) / max(1.0, max_esc)

                    max_g = float(self.grid_size * self.num_guards * 1.5)
                    closeness = (max_g - gdist) / max(1.0, max_g)  # 0 far -> 1 close
                    closeness = float(np.clip(closeness, 0.0, 1.0))
                    guard_penalty = -self.prisoner_guard_penalty_lambda * closeness
                    shaped = progress_reward + guard_penalty - self.prisoner_time_penalty

                self._prev_prisoner_escape_dist = esc_dist
                self._prev_prisoner_guard_dist = gdist
                rewards["prisoner"] = float(shaped)

        # --- truncation
        if self.timestep >= self.max_steps and not any(terminations.values()):
            truncations = {a: True for a in self.agents}
            outcome = "timeout"

        observations = {a: self._get_observation(a) for a in self.agents}
        for a in infos:
            infos[a]["episode_outcome"] = outcome

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return observations, rewards, terminations, truncations, infos

    # -------------------------
    # Spaces
    # -------------------------
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "prisoner":
            # prisoner_pos(2) + guards(2*num_guards) + escape_pos(2)
            dim = 2 + 2 * self.num_guards + 2
            return Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

        # guards: my_pos(2) + prisoner(2) + other_guards(2*(N-1)) + escape(2) + id_onehot(N)
        dim = 2 + 2 + 2 * (self.num_guards - 1) + 2 + self.num_guards
        return Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    # -------------------------
    # Render (same drawing style)
    # -------------------------
    def render(self):
        if self.render_mode is None:
            return

        if (self.render_mode in ["human", "rgb_array"]) and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if (self.render_mode in ["human", "rgb_array"]) and self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(None, int(self.cell_size / 2))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (0, 0, 0),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size), 1
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (0, 0, 0),
                (0, y * self.cell_size),
                (self.window_size, y * self.cell_size), 1
            )

        pygame.draw.rect(
            canvas, (255, 0, 0),
            pygame.Rect(
                self.escape_pos[0] * self.cell_size,
                self.escape_pos[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
        )

        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        pygame.draw.circle(
            canvas, (0, 0, 255),
            ((prisoner_pos[0] + 0.5) * self.cell_size, (prisoner_pos[1] + 0.5) * self.cell_size),
            self.cell_size / 3
        )

        for i in range(self.num_guards):
            guard_pos = self.agents_obj[f"guard_{i}"]["pos"]
            center_x = (guard_pos[0] + 0.5) * self.cell_size
            center_y = (guard_pos[1] + 0.5) * self.cell_size
            pygame.draw.circle(canvas, (0, 255, 0), (center_x, center_y), self.cell_size / 3)

            if self.font:
                text_surf = self.font.render(str(i), True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(center_x, center_y))
                canvas.blit(text_surf, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
