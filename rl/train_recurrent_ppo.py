import argparse
import os
import time

import httpx
import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage

from envs.emerald_http_env import EmeraldHTTPEnv


def make_env(server_url: str, reset_state: str | None, max_steps: int, action_repeat: int):
    def _thunk():
        return EmeraldHTTPEnv(
            server_url=server_url,
            reset_state_path=reset_state,
            max_steps_per_episode=max_steps,
            action_repeat=action_repeat,
        )
    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on Pokemon Emerald via HTTP server")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--reset-state", type=str, default="tests/states/simple_test.state")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--save-path", type=str, default="checkpoints/recurrent_ppo_emerald")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Check server availability
    try:
        r = httpx.get(f"{args.server_url.rstrip('/')}/status", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Cannot reach server at {args.server_url}. Start it with `python server/app.py` (error: {e})")

    # Single environment (the server is single-instance by default)
    env = DummyVecEnv([make_env(args.server_url, args.reset_state, args.max_steps, args.action_repeat)])
    # Convert HWC to CHW for CNN policies
    env = VecTransposeImage(env)

    # Recurrent CNN policy (handles image inputs and LSTM hidden states)
    model = RecurrentPPO(
        RecurrentActorCriticCnnPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=0.1,
        seed=args.seed,
        device="auto",
    )

    print("Starting training... Make sure the FastAPI server is running (server/app.py)")
    start = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.2f} min")

    model_path = os.path.join(args.save_path, f"model_{int(time.time())}")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()


