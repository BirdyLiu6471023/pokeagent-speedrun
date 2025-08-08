import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import httpx
import numpy as np
from PIL import Image


class EmeraldHTTPEnv(gym.Env):
    """
    Gymnasium environment that controls the existing FastAPI Emerald server via HTTP.

    Observation: 84x84 grayscale image (uint8)
    Action: Discrete set of single button presses + NOOP
    Reward: Shaped from server state deltas (money, pokedex_seen, badges, movement)
    Episode termination: time-limit based; optional early termination on large progress
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        reset_state_path: Optional[str] = None,
        frame_size: Tuple[int, int] = (84, 84),
        max_steps_per_episode: int = 200,
        reward_weights: Optional[Dict[str, float]] = None,
        action_repeat: int = 1,
        reset_milestones_on_reset: bool = True,
        request_timeout: float = 20.0,
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.reset_state_path = reset_state_path
        self.frame_w, self.frame_h = frame_size
        self.max_steps_per_episode = max_steps_per_episode
        self.action_repeat = max(1, int(action_repeat))
        self.reset_milestones_on_reset = reset_milestones_on_reset
        self.request_timeout = request_timeout

        # Discrete actions: NOOP + A,B,START,SELECT,UP,DOWN,LEFT,RIGHT
        self._action_buttons: List[List[str]] = [
            [],
            ["A"], ["B"], ["START"], ["SELECT"],
            ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],
        ]

        self.action_space = gym.spaces.Discrete(len(self._action_buttons))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.frame_h, self.frame_w, 1), dtype=np.uint8
        )

        # Reward weights
        self.reward_weights = {
            "money": 0.001,            # per money gained
            "pokedex_seen": 0.5,       # per new seen
            "pokedex_caught": 1.0,     # per new caught
            "badges": 50.0,            # per new badge
            "move": 0.01,              # slight reward for changing position
        }
        if reward_weights:
            self.reward_weights.update(reward_weights)

        self._client = httpx.Client(timeout=self.request_timeout)

        # Episode state
        self._step_count = 0
        self._last_state: Optional[Dict[str, Any]] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    # --------------- Helpers ---------------
    def _decode_obs(self, state: Dict[str, Any]) -> np.ndarray:
        vis = state.get("visual", {})
        b64 = vis.get("screenshot_base64")
        if not b64:
            # fallback to black frame
            return np.zeros((self.frame_h, self.frame_w, 1), dtype=np.uint8)
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("L")
        img = img.resize((self.frame_w, self.frame_h), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        return arr[:, :, None]

    def _get_state(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.server_url}/state")
        resp.raise_for_status()
        return resp.json()

    def _post_action(self, buttons: List[str]) -> Dict[str, Any]:
        payload = {"buttons": buttons}
        resp = self._client.post(f"{self.server_url}/action", json=payload)
        resp.raise_for_status()
        # The /action endpoint returns screenshot and status; we still call /state for full state
        return resp.json()

    def _reset_server(self) -> None:
        # Optionally clear milestones to avoid reward leakage between episodes
        if self.reset_milestones_on_reset:
            try:
                self._client.post(f"{self.server_url}/debug/reset_milestones")
            except Exception:
                pass
        # If a state path is provided, load it for consistent resets
        if self.reset_state_path:
            try:
                self._client.post(f"{self.server_url}/load_state", params={"filename": self.reset_state_path})
            except Exception as e:
                # Allow training to continue even if load fails
                print(f"[EmeraldHTTPEnv] Warning: load_state failed: {e}")

    def _extract_position(self, state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        pos = state.get("player", {}).get("position")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            try:
                return int(pos["x"]), int(pos["y"])
            except Exception:
                return None
        return None

    def _reward_from_delta(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> float:
        rew = 0.0
        # Money
        money_prev = prev.get("game", {}).get("money", 0) if prev else 0
        money_curr = curr.get("game", {}).get("money", 0)
        if isinstance(money_prev, int) and isinstance(money_curr, int):
            rew += (money_curr - money_prev) * self.reward_weights["money"]

        # Pokedex
        seen_prev = prev.get("game", {}).get("pokedex_seen", 0) if prev else 0
        seen_curr = curr.get("game", {}).get("pokedex_seen", 0)
        if isinstance(seen_prev, int) and isinstance(seen_curr, int):
            rew += (seen_curr - seen_prev) * self.reward_weights["pokedex_seen"]

        caught_prev = prev.get("game", {}).get("pokedex_caught", 0) if prev else 0
        caught_curr = curr.get("game", {}).get("pokedex_caught", 0)
        if isinstance(caught_prev, int) and isinstance(caught_curr, int):
            rew += (caught_curr - caught_prev) * self.reward_weights["pokedex_caught"]

        # Badges: support list or count
        badges_prev = prev.get("game", {}).get("badges", 0) if prev else 0
        badges_curr = curr.get("game", {}).get("badges", 0)
        def badge_count(x: Any) -> int:
            if isinstance(x, int):
                return x
            if isinstance(x, list):
                return sum(1 for b in x if b)
            return 0
        rew += (badge_count(badges_curr) - badge_count(badges_prev)) * self.reward_weights["badges"]

        # Movement reward (tiny)
        prev_pos = self._extract_position(prev) if prev else None
        curr_pos = self._extract_position(curr)
        if prev_pos and curr_pos and prev_pos != curr_pos:
            rew += self.reward_weights["move"]

        return float(rew)

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_count = 0
        self._last_state = None
        self._last_pos = None
        self._reset_server()

        # Small wait to let server settle after reset
        time.sleep(0.05)
        state = self._get_state()
        obs = self._decode_obs(state)
        self._last_state = state
        self._last_pos = self._extract_position(state)
        info = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        buttons = self._action_buttons[int(action)]
        # Repeat action for action_repeat frames
        for _ in range(self.action_repeat):
            self._post_action(buttons)

        state = self._get_state()
        obs = self._decode_obs(state)

        reward = self._reward_from_delta(self._last_state or {}, state)
        self._last_state = state
        curr_pos = self._extract_position(state)
        self._last_pos = curr_pos

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps_per_episode
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):  # type: ignore[override]
        # Return last observation (no separate render)
        if self._last_state is None:
            return None
        return self._decode_obs(self._last_state)

    def close(self):  # type: ignore[override]
        try:
            self._client.close()
        except Exception:
            pass


