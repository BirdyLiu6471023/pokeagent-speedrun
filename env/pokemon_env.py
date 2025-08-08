import os
from typing import Dict, Tuple, Any, Optional

import numpy as np

try:  # gym is optional; fall back to gymnasium if available
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - gymnasium fallback
    import gymnasium as gym
    from gymnasium import spaces

from pokemon_env.emulator import EmeraldEmulator


class PokemonEnv(gym.Env):
    """Gym-style environment wrapping the Emerald emulator.

    Observations are dictionaries containing:
    - ``screen``: RGB array of the current frame
    - ``party``: array with basic info about the player's party
    - ``position``: ``(x, y)`` map coordinates

    The environment currently exposes a discrete action space mapped to
    common Game Boy Advance buttons. Reward shaping is intentionally
    minimal and set to zero – users of this environment are expected to
    provide their own reward functions during training.
    """

    metadata = {"render.modes": ["rgb_array"]}

    ACTIONS = ["A", "B", "LEFT", "RIGHT", "UP", "DOWN", "START", "SELECT", "L", "R"]

    def __init__(self, rom_path: str, initial_state: Optional[str] = None):
        super().__init__()
        self.emulator = EmeraldEmulator(rom_path)
        self.state_path = initial_state
        if initial_state:
            # Load provided state on creation so that reset is cheap
            self.emulator.load_state(initial_state)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # Observation components
        screen_shape = (240, 160, 3)  # GBA resolution
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8),
                "party": spaces.Box(low=0, high=65535, shape=(6, 3), dtype=np.int32),
                "position": spaces.Box(low=0, high=65535, shape=(2,), dtype=np.int32),
            }
        )
        self._last_obs: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_screen(self) -> np.ndarray:
        frame = self.emulator.get_latest_frame()
        if frame is None:
            shot = self.emulator.get_screenshot()
            frame = np.array(shot) if shot is not None else np.zeros((240, 160, 3), dtype=np.uint8)
        return frame

    def _get_party(self) -> np.ndarray:
        party_arr = np.zeros((6, 3), dtype=np.int32)
        reader = getattr(self.emulator, "memory_reader", None)
        if reader is not None:
            try:
                party = reader.read_party_pokemon()
                for i, pkmn in enumerate(party[:6]):
                    party_arr[i, 0] = pkmn.get("species", 0)
                    party_arr[i, 1] = pkmn.get("current_hp", 0)
                    party_arr[i, 2] = pkmn.get("max_hp", 0)
            except Exception:
                pass
        return party_arr

    def _get_position(self) -> np.ndarray:
        coords = (0, 0)
        reader = getattr(self.emulator, "memory_reader", None)
        if reader is not None:
            try:
                coords = reader.read_coordinates()
            except Exception:
                pass
        return np.array(coords, dtype=np.int32)

    def _get_obs(self) -> Dict[str, Any]:
        obs = {
            "screen": self._get_screen(),
            "party": self._get_party(),
            "position": self._get_position(),
        }
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        state_file = options.get("state_path") if options else self.state_path
        if state_file:
            self.emulator.load_state(state_file)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        button = self.ACTIONS[action]
        # Advance one frame with the chosen button held
        self.emulator.run_frame_with_buttons([button])
        obs = self._get_obs()
        reward = 0.0  # Placeholder reward
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only rgb_array rendering is supported")
        return self._get_screen()

    def close(self):  # pragma: no cover - clean up emulator resources
        try:
            if self.emulator:
                self.emulator.shutdown()
        except Exception:
            pass
