import functools
import random
from copy import copy

import numpy as np
import pygame
from gymnasium.spaces import Box
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
        grid_size=100,
        num_guards=2,
        max_steps=100,
        seed=None,
        training_side="guards",
        # Opponent policies
        guard_model_path=None,
        prisoner_model_path=None,
        prisoner_use_heuristic=True,
        # Continuous world params
        max_speed=2.0,
        max_accel=1.0,
        capture_radius=0.3,
        escape_radius=0.3,
        # Reward shaping weights
        guard_escape_penalty_lambda=0.05,
        guard_time_penalty=0.01,
        prisoner_guard_penalty_lambda=0.05,
        prisoner_time_penalty=0.01,
    ):
        self.render_mode = render_mode
        self.grid_size = float(grid_size)
        self.num_guards = int(num_guards)
        self.max_steps = int(max_steps)
        self.training_side = training_side

        self.guard_model_path = guard_model_path
        self.prisoner_model_path = prisoner_model_path
        self.prisoner_use_heuristic = prisoner_use_heuristic

        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.capture_radius = float(capture_radius)
        self.escape_radius = float(escape_radius)

        self.guard_escape_penalty_lambda = float(guard_escape_penalty_lambda)
        self.guard_time_penalty = float(guard_time_penalty)
        self.prisoner_guard_penalty_lambda = float(prisoner_guard_penalty_lambda)
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
        self.escape_pos = np.zeros(2, dtype=np.float32)
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
    def _apply_accel(self, pos: np.ndarray, vel: np.ndarray, accel: np.ndarray) -> None:
        a = np.array(accel, dtype=np.float32)
        if a.shape != (2,):
            a = a.reshape(2,)
        a = np.clip(a, -1.0, 1.0)
        # scale to max acceleration and clamp magnitude
        a = a * self.max_accel
        a_norm = float(np.linalg.norm(a))
        if a_norm > self.max_accel and a_norm > 0:
            a = a * (self.max_accel / a_norm)

        vel[:] = vel + a
        v_norm = float(np.linalg.norm(vel))
        if v_norm > self.max_speed and v_norm > 0:
            vel[:] = vel * (self.max_speed / v_norm)

        pos[:] = pos + vel
        pos[:] = np.clip(pos, 0.0, self.grid_size)

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

    # -------------------------
    # Observations (A)
    # -------------------------
    def _norm(self, v: np.ndarray, denom: float) -> np.ndarray:
        denom = float(max(1e-6, denom))
        return v.astype(np.float32) / denom

    def _get_guard_obs(self, guard_id: str) -> np.ndarray:
        """A (guards): my_pos + prisoner_pos + other_guards + escape_pos + id_onehot"""
        k = int(guard_id.split("_")[1])

        my_pos = self._norm(self.agents_obj[guard_id]["pos"], self.grid_size)
        my_vel = self._norm(self.agents_obj[guard_id]["vel"], self.max_speed)
        prisoner_pos = self._norm(self.agents_obj["prisoner"]["pos"], self.grid_size)
        prisoner_vel = self._norm(self.agents_obj["prisoner"]["vel"], self.max_speed)
        escape_pos = self._norm(self.escape_pos, self.grid_size)

        others = []
        for i in range(self.num_guards):
            if i == k:
                continue
            others.append(self._norm(self.agents_obj[f"guard_{i}"]["pos"], self.grid_size))
            others.append(self._norm(self.agents_obj[f"guard_{i}"]["vel"], self.max_speed))
        others = np.concatenate(others) if others else np.array([], dtype=np.float32)

        id_onehot = np.zeros(self.num_guards, dtype=np.float32)
        id_onehot[k] = 1.0

        return np.concatenate(
            [my_pos, my_vel, prisoner_pos, prisoner_vel, others, escape_pos, id_onehot]
        ).astype(np.float32)

    def _get_prisoner_obs(self) -> np.ndarray:
        """A (prisoner): prisoner_pos + all_guards_pos + escape_pos"""
        prisoner_pos = self._norm(self.agents_obj["prisoner"]["pos"], self.grid_size)
        prisoner_vel = self._norm(self.agents_obj["prisoner"]["vel"], self.max_speed)
        guards = []
        for i in range(self.num_guards):
            guards.append(self._norm(self.agents_obj[f"guard_{i}"]["pos"], self.grid_size))
            guards.append(self._norm(self.agents_obj[f"guard_{i}"]["vel"], self.max_speed))
        guards = np.concatenate(guards) if guards else np.array([], dtype=np.float32)
        escape_pos = self._norm(self.escape_pos, self.grid_size)
        return np.concatenate([prisoner_pos, prisoner_vel, guards, escape_pos]).astype(np.float32)

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

    def _prisoner_heuristic_action(self) -> np.ndarray:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        escape_pos = self.escape_pos
        guards = [self.agents_obj[f"guard_{i}"]["pos"] for i in range(self.num_guards)]
        if not guards:
            return np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

        distances = [np.linalg.norm(prisoner_pos - g) for g in guards]
        closest_guard = guards[int(np.argmin(distances))]

        escape_vec = escape_pos - prisoner_pos
        avoid_vec = prisoner_pos - closest_guard
        final_vec = 2 * escape_vec + avoid_vec

        norm = float(np.linalg.norm(final_vec))
        if norm == 0:
            return np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        return (final_vec / norm).astype(np.float32)

    def _get_prisoner_action(self) -> np.ndarray:
        if self.training_side == "play":
            raise RuntimeError("play mode expects prisoner action from outside")
        if self.prisoner_use_heuristic:
            return self._prisoner_heuristic_action()
        self._lazy_load_prisoner_model()
        obs = self._get_prisoner_obs()
        a, _ = self._prisoner_model.predict(obs, deterministic=True)
        return np.array(a, dtype=np.float32)

    def _get_guard_actions_from_model(self) -> dict:
        self._lazy_load_guard_model()
        actions = {}
        for i in range(self.num_guards):
            gid = f"guard_{i}"
            obs = self._get_guard_obs(gid)
            a, _ = self._guard_model.predict(obs, deterministic=True)
            actions[gid] = np.array(a, dtype=np.float32)
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
            self.agents_obj = {
                "prisoner": {
                    "role": "prisoner",
                    "pos": np.array([0.0, 0.0], dtype=np.float32),
                    "vel": np.zeros(2, dtype=np.float32),
                }
            }
            for i in range(self.num_guards):
                self.agents_obj[f"guard_{i}"] = {
                    "role": "guard",
                    "pos": np.array(
                        [random.uniform(1.0, self.grid_size), random.uniform(1.0, self.grid_size)],
                        dtype=np.float32,
                    ),
                    "vel": np.zeros(2, dtype=np.float32),
                }
            self.escape_pos = np.array(
                [random.uniform(0.0, self.grid_size), random.uniform(0.0, self.grid_size)],
                dtype=np.float32,
            )

            prisoner_pos = self.agents_obj["prisoner"]["pos"]
            too_close_guard = any(
                float(np.linalg.norm(prisoner_pos - self.agents_obj[f"guard_{i}"]["pos"])) <= self.capture_radius
                for i in range(self.num_guards)
            )
            too_close_escape = (float(np.linalg.norm(prisoner_pos - self.escape_pos)) <= self.escape_radius)

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
                full_actions[gid] = np.array(actions.get(gid, self.action_space(gid).sample()), dtype=np.float32)
            full_actions["prisoner"] = self._get_prisoner_action()

        elif self.training_side == "prisoner":
            # prisoner from outside, guards from fixed guard model
            full_actions["prisoner"] = np.array(
                actions.get("prisoner", self.action_space("prisoner").sample()),
                dtype=np.float32,
            )
            full_actions.update(self._get_guard_actions_from_model())

        elif self.training_side == "play":
            # both provided
            for i in range(self.num_guards):
                gid = f"guard_{i}"
                full_actions[gid] = np.array(actions.get(gid, self.action_space(gid).sample()), dtype=np.float32)
            full_actions["prisoner"] = np.array(
                actions.get("prisoner", self.action_space("prisoner").sample()),
                dtype=np.float32,
            )

        # --- apply moves
        self._apply_accel(
            self.agents_obj["prisoner"]["pos"],
            self.agents_obj["prisoner"]["vel"],
            full_actions["prisoner"],
        )
        for i in range(self.num_guards):
            gid = f"guard_{i}"
            self._apply_accel(
                self.agents_obj[gid]["pos"],
                self.agents_obj[gid]["vel"],
                full_actions[gid],
            )

        prisoner_pos = self.agents_obj["prisoner"]["pos"]

        captured = any(
            float(np.linalg.norm(prisoner_pos - self.agents_obj[f"guard_{i}"]["pos"])) <= self.capture_radius
            for i in range(self.num_guards)
        )
        escaped = (float(np.linalg.norm(prisoner_pos - self.escape_pos)) <= self.escape_radius)

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
                # Guards shaping: encourage decreasing total guard distance
                total_guard_dist = self._total_guard_distance_l2()
                dist_delta = self._prev_total_guard_dist - total_guard_dist
                self._prev_total_guard_dist = total_guard_dist

                max_ref = float(self.grid_size * self.num_guards * 1.5)
                approach_reward = float(dist_delta) / max(1.0, max_ref)

                # Penalize prisoner approaching escape
                esc_dist = self._escape_distance_l2()
                max_esc = float(np.sqrt(2) * (self.grid_size))
                escape_progress = (max_esc - esc_dist) / max(1.0, max_esc)
                escape_progress = float(np.clip(escape_progress, 0.0, 1.0))
                escape_penalty = -self.guard_escape_penalty_lambda * escape_progress

                shaped = approach_reward + escape_penalty - self.guard_time_penalty
                for a in rewards:
                    rewards[a] = float(shaped)

            elif self.training_side == "prisoner":
                # Prisoner shaping: encourage decreasing escape distance
                esc_dist = self._escape_distance_l2()
                delta = self._prev_prisoner_escape_dist - esc_dist
                self._prev_prisoner_escape_dist = esc_dist

                max_esc = float(np.sqrt(2) * (self.grid_size))
                progress_reward = float(delta) / max(1.0, max_esc)

                # Penalize being close to guards (based on total L2 distance)
                gdist = self._total_guard_distance_l2()
                max_g = float(self.grid_size * self.num_guards * 1.5)
                closeness = (max_g - gdist) / max(1.0, max_g)  # 0 far -> 1 close
                closeness = float(np.clip(closeness, 0.0, 1.0))
                guard_penalty = -self.prisoner_guard_penalty_lambda * closeness

                shaped = progress_reward + guard_penalty - self.prisoner_time_penalty
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
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent == "prisoner":
            # prisoner_pos(2) + prisoner_vel(2) + guards(pos+vel)(4*num_guards) + escape_pos(2)
            dim = 2 + 2 + 4 * self.num_guards + 2
            return Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

        # guards: my_pos(2) + my_vel(2) + prisoner(pos+vel)(4)
        # + other_guards(pos+vel)(4*(N-1)) + escape(2) + id_onehot(N)
        dim = 2 + 2 + 4 + 4 * (self.num_guards - 1) + 2 + self.num_guards
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
        canvas.fill((245, 245, 245))

        # Draw boundary
        pygame.draw.rect(canvas, (220, 220, 220), pygame.Rect(0, 0, self.window_size, self.window_size), 2)

        esc_r = max(3, int(self.escape_radius * self.cell_size))
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (int(self.escape_pos[0] * self.cell_size), int(self.escape_pos[1] * self.cell_size)),
            esc_r,
        )

        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        pygame.draw.circle(
            canvas, (0, 0, 255),
            (int(prisoner_pos[0] * self.cell_size), int(prisoner_pos[1] * self.cell_size)),
            max(4, int(self.cell_size / 3))
        )

        for i in range(self.num_guards):
            guard_pos = self.agents_obj[f"guard_{i}"]["pos"]
            center_x = guard_pos[0] * self.cell_size
            center_y = guard_pos[1] * self.cell_size
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (int(center_x), int(center_y)),
                max(4, int(self.cell_size / 3)),
            )

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
