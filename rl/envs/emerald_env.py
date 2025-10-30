"""
Gymnasium environment wrapper for the Pokemon Emerald HTTP server.

Observation: 84x84 grayscale image (uint8, channels-last)
Action space: Discrete buttons (NOOP, A, B, START, SELECT, UP, DOWN, LEFT, RIGHT)
Reward: Derived from JSON state deltas between steps (movement, progress, penalties)

This environment talks to the running FastAPI server in `server/app.py` via HTTP.
It does not access the emulator or memory reader directly.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover - guard for environments without gym
    raise RuntimeError(
        "Gymnasium is required for rl/envs/emerald_env.py. Install with: pip install gymnasium"
    ) from e


# Discrete action set used by the server/buttons
# Note: Keep this in sync with the server's expected button names
BUTTONS: Tuple[str, ...] = (
    "NOOP",
    "A",
    "B",
    "START",
    "SELECT",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
)


@dataclass
class EnvConfig:
    server_url: str = "http://127.0.0.1:8000"
    # When resetting, which state to load on the server (relative to repo root)
    reset_state_path: Optional[str] = "Emerald-GBAdvance/start.state"
    request_timeout_s: float = 10.0
    max_request_retries: int = 3
    retry_backoff_s: float = 0.5
    action_repeat: int = 2
    # Episode control
    max_episode_steps: int = 1000


def _b64_png_to_gray84(obs_png_b64: str) -> np.ndarray:
    if not obs_png_b64:
        # Return a black frame if missing
        return np.zeros((84, 84, 1), dtype=np.uint8)

    data = base64.b64decode(obs_png_b64)
    from PIL import Image  # Lazy import to avoid hard dep for non-RL workflows

    img = Image.open(io.BytesIO(data)).convert("L")  # grayscale
    img = img.resize((84, 84))
    arr = np.asarray(img, dtype=np.uint8)
    return arr[..., None]  # HWC with channel dim


def _safe_get(session: requests.Session, url: str, timeout: float, retries: int, backoff: float) -> Optional[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:  # noqa: PERF203 - small retry helper
            last_err = e
        time.sleep(backoff * (1 + attempt))
    if last_err:
        raise last_err
    return None


def _safe_post(session: requests.Session, url: str, json: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    try:
            resp = session.post(url, json=json, timeout=timeout)
            if resp.status_code == 200:
                # Some endpoints return no body
                try:
                    return resp.json()
                except Exception:
                    return {}
    except Exception as e:
        last_err = e

    if last_err:
        raise last_err
    return None


class EmeraldGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Observation: 84x84 grayscale, channels-last
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(BUTTONS))

        self._session = requests.Session()
        self._last_state: Optional[Dict[str, Any]] = None
        self._last_frame_b64: str = ""
        self._episode_steps: int = 0

    # --- Gymnasium API ---

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._episode_steps = 0

        # On reset, optionally load a known state via the server
        if self.config.reset_state_path:
            _safe_post(
                self._session,
                f"{self.config.server_url}/load_state",
                json={"filepath": self.config.reset_state_path},
                timeout=5
            )

        # Also reset server-side metrics to keep cleaner logs
        try:
            _safe_post(
                self._session,
                f"{self.config.server_url}/reset_metrics",
                json={},
                timeout=5
            )
        except Exception:
            # Non-fatal if not available
            pass

        obs, st = self._get_obs_and_state()
        self._last_state = st
        return obs, {"server_state": st}

    def step(self, action: int):  # type: ignore[override]
        assert self.action_space.contains(action)

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # Map discrete action to button(s)
        button = BUTTONS[action]
        buttons: Tuple[str, ...]
        if button == "NOOP":
            buttons = tuple()
        else:
            buttons = (button,)

        # Repeat the action for action_repeat steps
        for repeat_index in range(max(1, int(self.config.action_repeat))):
            if buttons:
                try:
                    print(f"[RL Env] Sending action (repeat {repeat_index+1}): {list(buttons)}")
                    response = requests.post(
                                                    f"{self.config.server_url}/action",
                                                    json={"buttons": buttons},
                                                    timeout=5
                                                )
                    if response.status_code == 200:
                        print(f"[RL Env] Action acknowledged by server {buttons}")
                    else:
                        print(f"[RL Env] Action send returned no response (possible non-200): {list(buttons)}")
                except Exception as e:
                    print(f"[RL Env] Action send FAILED: {list(buttons)} | Error: {e}")
            # Give the emulator a brief moment to advance
            time.sleep(0.05)

            obs, st = self._get_obs_and_state()
            reward = self._compute_reward(self._last_state, st)
            total_reward += reward
            self._last_state = st
            info = {"server_state": st, "reward_components": self._last_reward_components}

        self._episode_steps += 1

        if self._episode_steps >= self.config.max_episode_steps:
            truncated = True

        return obs, float(total_reward), terminated, truncated, info

    def render(self):  # type: ignore[override]
        # Return last RGB frame if available
        if not self._last_frame_b64:
            return None
        # Decode to RGB array for gym-compatible render
        data = base64.b64decode(self._last_frame_b64)
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def close(self):  # type: ignore[override]
        try:
            self._session.close()
        except Exception:
            pass

    # --- Helpers ---

    def _get_obs_and_state(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Prefer comprehensive state for reward; contains embedded screenshot
        st = _safe_get(
            self._session,
            f"{self.config.server_url}/state",
            timeout=self.config.request_timeout_s,
            retries=self.config.max_request_retries,
            backoff=self.config.retry_backoff_s,
        )
        if not isinstance(st, dict):
            st = {}

        visual = st.get("visual", {}) if isinstance(st, dict) else {}
        self._last_frame_b64 = str(visual.get("screenshot_base64", "")) if isinstance(visual, dict) else ""

        obs = _b64_png_to_gray84(self._last_frame_b64)
        return obs, st

    def _compute_reward(self, prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> float:
        # Reward from deltas; store components for debugging
        self._last_reward_components = {}
        if not prev or not isinstance(prev, dict) or not isinstance(cur, dict):
            self._last_reward_components = {"bootstrap": 0.0}
            return 0.0

        reward = 0.0

        # - Small living penalty to encourage faster progress
        living_penalty = -0.01
        reward += living_penalty
        self._last_reward_components["living_penalty"] = living_penalty

        # Movement reward: +0.1 per tile moved (Manhattan distance)
        try:
            ppos = prev.get("player", {}).get("position", {})
            cpos = cur.get("player", {}).get("position", {})
            if ppos and cpos:
                dx = abs(int(cpos.get("x", 0)) - int(ppos.get("x", 0)))
                dy = abs(int(cpos.get("y", 0)) - int(ppos.get("y", 0)))
                move_r = 0.1 * float(dx + dy)
                reward += move_r
                self._last_reward_components["movement"] = move_r
        except Exception:
            pass

        # Badge progress: large positive
        try:
            p_badges = int(prev.get("game", {}).get("badges", 0))
            c_badges = int(cur.get("game", {}).get("badges", 0))
            if c_badges > p_badges:
                badge_r = 50.0 * float(c_badges - p_badges)
                reward += badge_r
                self._last_reward_components["badges"] = badge_r
        except Exception:
            pass

        # Money increase: small positive
        try:
            p_money = int(prev.get("game", {}).get("money", 0))
            c_money = int(cur.get("game", {}).get("money", 0))
            if c_money > p_money:
                money_r = 0.001 * float(c_money - p_money)
                reward += money_r
                self._last_reward_components["money"] = money_r
        except Exception:
            pass

        # Dialog penalty to discourage being stuck in menus/dialog
        try:
            p_state = str(prev.get("game", {}).get("game_state", ""))
            c_state = str(cur.get("game", {}).get("game_state", ""))
            if c_state.lower() in {"dialog", "menu"} and p_state.lower() == c_state.lower():
                dialog_penalty = -0.02
                reward += dialog_penalty
                self._last_reward_components["dialog_penalty"] = dialog_penalty
        except Exception:
            pass

        return float(reward)


