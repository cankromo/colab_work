import functools
import random
from copy import copy

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv


class CustomEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "custom_environment_v0",
        "render_fps": 4,
    }

    def __init__(self, render_mode=None, grid_size=10, num_guards=2, max_steps=100, seed=None):
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.num_guards = num_guards
        self.max_steps = max_steps

        # Guards-only learning agents (parameter sharing hedefin)
        self.possible_agents = [f"guard_{i}" for i in range(self.num_guards)]
        self.agents = []

        self.agents_obj = {}
        self.escape_pos = np.zeros(2, dtype=np.int32)
        self.timestep = 0

        # Shaping için önceki mesafeler
        self._prev_total_guard_dist = None
        self._prev_escape_dist = None

        # Render alanları
        self.window = None
        self.clock = None
        self.font = None

        if self.render_mode in ["human", "rgb_array"]:
            self.window_size = 512
            self.cell_size = self.window_size / self.grid_size

        # Escape penalty ağırlığı (küçük tut!)
        self.escape_penalty_lambda = 0.05
        # Time penalty
        self.time_penalty = 0.01

        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # -------------------------
    # Prisoner heuristic policy
    # -------------------------
    def _get_prisoner_action(self) -> int:
        """Mahkum için kural tabanlı aksiyon (öğrenmiyor)."""
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        escape_pos = self.escape_pos

        guards = [self.agents_obj[f"guard_{i}"]["pos"] for i in range(self.num_guards)]
        if not guards:
            return random.randint(0, 3)

        distances = [np.linalg.norm(prisoner_pos - g) for g in guards]
        closest_guard = guards[int(np.argmin(distances))]

        escape_vec = escape_pos - prisoner_pos
        avoid_vec = prisoner_pos - closest_guard

        # Kaçışa öncelik: 2*escape + avoid
        final_vec = 2 * escape_vec + avoid_vec

        if abs(final_vec[0]) > abs(final_vec[1]):
            return 1 if final_vec[0] > 0 else 0
        else:
            return 3 if final_vec[1] > 0 else 2

    # -------------------------
    # Movement + helpers
    # -------------------------
    def _apply_move(self, pos: np.ndarray, action: int, move_count: int = 1) -> None:
        for _ in range(move_count):
            # x axis
            if action == 0 and pos[0] > 0:
                pos[0] -= 1
            elif action == 1 and pos[0] < self.grid_size - 1:
                pos[0] += 1
            # y axis
            elif action == 2 and pos[1] > 0:
                pos[1] -= 1
            elif action == 3 and pos[1] < self.grid_size - 1:
                pos[1] += 1

    def _chebyshev_dist(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.linalg.norm(a - b, ord=np.inf))

    def _total_guard_distance_l2(self) -> float:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        total = 0.0
        for i in range(self.num_guards):
            gpos = self.agents_obj[f"guard_{i}"]["pos"]
            total += float(np.linalg.norm(prisoner_pos - gpos))  # L2
        return total

    def _escape_distance_l2(self) -> float:
        prisoner_pos = self.agents_obj["prisoner"]["pos"]
        return float(np.linalg.norm(prisoner_pos - self.escape_pos))  # L2

    def _sample_initial_state(self):
        # Prisoner fixed start
        self.agents_obj = {
            "prisoner": {"role": "prisoner", "pos": np.array([0, 0], dtype=np.int32)}
        }
        # Guards random
        for i in range(self.num_guards):
            gid = f"guard_{i}"
            self.agents_obj[gid] = {
                "role": "guard",
                "pos": np.array(
                    [random.randint(1, self.grid_size - 1), random.randint(1, self.grid_size - 1)],
                    dtype=np.int32,
                ),
            }
        # Escape random
        self.escape_pos = np.array(
            [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)],
            dtype=np.int32,
        )

    # -------------------------
    # PettingZoo API
    # -------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Başlangıç state (instant terminal’i azaltmak için dene-üret)
        for _ in range(200):
            self._sample_initial_state()
            prisoner_pos = self.agents_obj["prisoner"]["pos"]

            # prisoner guard’a çok yakın başlama
            too_close_guard = any(
                self._chebyshev_dist(prisoner_pos, self.agents_obj[f"guard_{i}"]["pos"]) <= 1
                for i in range(self.num_guards)
            )
            # prisoner escape’e çok yakın başlama
            too_close_escape = (self._chebyshev_dist(prisoner_pos, self.escape_pos) <= 1)

            if (not too_close_guard) and (not too_close_escape):
                break

        # Shaping baseline
        self._prev_total_guard_dist = self._total_guard_distance_l2()
        self._prev_escape_dist = self._escape_distance_l2()

        observations = {a: self._get_observation(a) for a in self.agents}
        infos = {a: {"episode_outcome": "running"} for a in self.agents}

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return observations, infos

    def step(self, actions):
        """
        actions: dict { "guard_0": int, "guard_1": int, ... }
        prisoner action env içinde heuristic olarak üretilir.
        """
        # 1) timestep
        self.timestep += 1

        # 2) Prisoner move (2x)  ✅ senin hedefin buydu
        prisoner_action = self._get_prisoner_action()
        self._apply_move(self.agents_obj["prisoner"]["pos"], prisoner_action, move_count=2)

        # 3) Guards move (1x)
        for i in range(self.num_guards):
            gid = f"guard_{i}"

            # ✅ SuperSuit bazı adımlarda bazı ajanları actions dict’ine koymayabilir
            if gid in actions:
                act = int(actions[gid])
            else:
                act = self.action_space(gid).sample()

            self._apply_move(self.agents_obj[gid]["pos"], act, move_count=1)

        # 4) Terminal checks
        prisoner_pos = self.agents_obj["prisoner"]["pos"]

        captured = any(
            self._chebyshev_dist(prisoner_pos, self.agents_obj[f"guard_{i}"]["pos"]) <= 1
            for i in range(self.num_guards)
        )
        escaped = (self._chebyshev_dist(prisoner_pos, self.escape_pos) <= 1)

        # 5) Outputs
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        infos = {a: {} for a in self.agents}

        # 6) Reward logic
        if captured:
            for a in rewards:
                rewards[a] = 1.0
                terminations[a] = True

            outcome = "captured"

        elif escaped:
            for a in rewards:
                rewards[a] = -1.0
                terminations[a] = True

            outcome = "escaped"

        else:
            # (A) Guards approach shaping: reward if total guard distance decreases
            total_guard_dist = self._total_guard_distance_l2()
            dist_delta = (self._prev_total_guard_dist - total_guard_dist)
            self._prev_total_guard_dist = total_guard_dist

            max_ref = float(self.grid_size * self.num_guards * 1.5)
            approach_reward = float(dist_delta) / max(1.0, max_ref)

            # (B) Escape-progress penalty: prisoner escape'e yaklaşıyorsa ceza
            escape_dist = self._escape_distance_l2()
            max_escape_dist = float(np.sqrt(2) * (self.grid_size - 1))

            escape_progress = (max_escape_dist - escape_dist) / max(1.0, max_escape_dist)
            escape_progress = float(np.clip(escape_progress, 0.0, 1.0))

            escape_penalty = -self.escape_penalty_lambda * escape_progress

            # (C) Time penalty
            time_pen = -self.time_penalty

            shaped_total = approach_reward + escape_penalty + time_pen

            for a in rewards:
                rewards[a] = float(shaped_total)

            outcome = "running"

        # 7) Truncation (time limit)
        if self.timestep >= self.max_steps and not any(terminations.values()):
            truncations = {a: True for a in self.agents}
            outcome = "timeout"

        # 8) Observations
        observations = {a: self._get_observation(a) for a in self.agents}

        # 9) infos outcome (callback bunu okuyacak)
        for a in infos:
            infos[a]["episode_outcome"] = outcome

        # 10) Episode end
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        # 11) Render
        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return observations, rewards, terminations, truncations, infos

    # -------------------------
    # Spaces + observations
    # -------------------------
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # my_pos(2) + prisoner_pos(2) + other_guards(2*(N-1)) + escape_pos(2) + id_onehot(N)
        dim = 2 + 2 + 2 * (self.num_guards - 1) + 2 + self.num_guards
        return Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _get_observation(self, agent):
        k = int(agent.split("_")[1])

        my_pos = self.agents_obj[agent]["pos"].astype(np.float32)
        prisoner_pos = self.agents_obj["prisoner"]["pos"].astype(np.float32)
        escape_pos = self.escape_pos.astype(np.float32)

        others = []
        for i in range(self.num_guards):
            if i == k:
                continue
            others.append(self.agents_obj[f"guard_{i}"]["pos"].astype(np.float32))
        others = np.concatenate(others) if others else np.array([], dtype=np.float32)

        id_onehot = np.zeros(self.num_guards, dtype=np.float32)
        id_onehot[k] = 1.0

        # normalize coords -> [0,1]
        denom = float(max(1, self.grid_size - 1))
        my_pos /= denom
        prisoner_pos /= denom
        escape_pos /= denom
        if others.size > 0:
            others = others / denom

        obs = np.concatenate([my_pos, prisoner_pos, others, escape_pos, id_onehot]).astype(np.float32)
        return obs

    # -------------------------
    # Render (SENİN KODUN - DEĞİŞMEDİ)
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