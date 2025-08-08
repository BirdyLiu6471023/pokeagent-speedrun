"""Training script for PPO-LSTM agent on PokemonEnv."""

import argparse
import logging
import os

from env.pokemon_env import PokemonEnv
from rl.ppo_lstm_agent import PPOLSTMAgent


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-LSTM agent on Pokemon Emerald")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/Emerald.gba", help="Path to ROM file")
    parser.add_argument("--state", type=str, default=None, help="Initial savestate path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    env = PokemonEnv(args.rom, args.state)
    agent = PPOLSTMAgent(env.observation_space, env.action_space, lr=args.lr, gamma=args.gamma)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        agent.reset_hidden()
        total_reward = 0.0
        for step in range(args.max_steps):
            action, value, log_prob, obs_t = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs_t, action, reward, value, log_prob, done)
            total_reward += reward
            obs = next_obs
            if done:
                break
        agent.learn()
        logger.info("Episode %d: reward=%.2f steps=%d", ep, total_reward, step + 1)
        if ep % args.checkpoint_interval == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"ppo_lstm_ep{ep}.pt")
            agent.save(ckpt)
            logger.info("Saved checkpoint to %s", ckpt)
    env.close()


if __name__ == "__main__":
    main()
