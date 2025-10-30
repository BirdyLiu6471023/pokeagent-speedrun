"""
Recurrent PPO training entrypoint for Pokemon Emerald via Gymnasium env.

Usage (from repo root):
  uv run python -m rl.train_recurrent_ppo --server http://127.0.0.1:8000 --timesteps 10000

Ensure the FastAPI server (server/app.py) is running and the emulator is active.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from rl.envs import EmeraldGymEnv, EnvConfig


def make_env(server_url: str, reset_state: Optional[str], action_repeat: int, max_steps: int) -> EmeraldGymEnv:
    config = EnvConfig(
        server_url=server_url,
        reset_state_path=reset_state,
        action_repeat=action_repeat,
        max_episode_steps=max_steps,
    )
    return EmeraldGymEnv(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--reset-state", type=str, default="Emerald-GBAdvance/start.state", help="State file path for reset")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--action-repeat", type=int, default=2, help="Repeat each action this many frames")
    parser.add_argument("--episode-steps", type=int, default=1000, help="Max steps per episode (env truncation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=".pokeagent_cache/ppo_emerald_lstm")
    args = parser.parse_args()

    # Lazy import SB3 to avoid forcing dependency for non-RL workflows
    try:
        from sb3_contrib import RecurrentPPO
        from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This script requires stable-baselines3 and sb3-contrib.\n"
            "Install with: pip install gymnasium stable-baselines3 sb3-contrib"
        ) from e

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    def _env_fn():
        return make_env(args.server, args.reset_state, args.action_repeat, args.episode_steps)

    # Vectorize even single env for compatibility
    vec_env = DummyVecEnv([_env_fn])

    model = RecurrentPPO(
        policy=RecurrentActorCriticCnnPolicy,
        env=vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(os.path.dirname(args.save_path), "tb"),
        n_steps=256,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        n_epochs=4,
    )

    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=args.save_path, name_prefix="rppo")

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True)
    model.save(args.save_path)


if __name__ == "__main__":
    main()


