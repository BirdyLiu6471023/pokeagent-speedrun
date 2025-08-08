"""Simple PPO agent with an LSTM policy layer.

This implementation is intentionally lightweight – it is **not** a
production-ready RL algorithm but provides the scaffolding required to
train and run a recurrent policy for the Pokemon environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    import gym
except Exception:  # pragma: no cover
    import gymnasium as gym


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    reward: float
    value: torch.Tensor
    log_prob: torch.Tensor
    done: bool


class LSTMPolicy(nn.Module):
    """Small LSTM-based actor-critic network."""

    def __init__(self, obs_size: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(obs_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(self, obs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], deterministic: bool = False):
        out, hidden = self.lstm(obs, hidden)
        feat = out[:, -1]
        logits = self.actor(feat)
        value = self.critic(feat)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, log_prob, hidden, dist

    def evaluate_actions(self, obs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], actions: torch.Tensor):
        out, hidden = self.lstm(obs, hidden)
        feat = out[:, -1]
        logits = self.actor(feat)
        value = self.critic(feat)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


class PPOLSTMAgent:
    """Lightweight PPO-style agent with LSTM hidden state management."""

    action_list = ["A", "B", "LEFT", "RIGHT", "UP", "DOWN", "START", "SELECT", "L", "R"]

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr: float = 3e-4, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Observation is processed into a simple vector: mean pixel, party info, position
        self.obs_size = 1
        if isinstance(observation_space, gym.spaces.Dict):
            party_shape = observation_space["party"].shape
            pos_shape = observation_space["position"].shape
            self.obs_size += int(np.prod(party_shape)) + int(np.prod(pos_shape))
        self.policy = LSTMPolicy(self.obs_size, action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.memory: List[Transition] = []
        self.reset_hidden()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs: Dict[str, Any]) -> torch.Tensor:
        screen_mean = np.array([obs["screen"].mean() / 255.0], dtype=np.float32)
        party = obs["party"].astype(np.float32).flatten() / 255.0
        position = obs["position"].astype(np.float32) / 255.0
        vec = np.concatenate([screen_mean, party, position]).astype(np.float32)
        tensor = torch.tensor(vec, device=self.device).unsqueeze(0).unsqueeze(0)
        return tensor

    def reset_hidden(self, batch_size: int = 1):
        self.hidden = self.policy.init_hidden(batch_size, self.device)

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any], deterministic: bool = False):
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            action, value, log_prob, self.hidden, _ = self.policy(obs_t, self.hidden, deterministic)
        return action.item(), value, log_prob, obs_t

    def store_transition(self, obs: torch.Tensor, action: int, reward: float, value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        self.memory.append(Transition(obs, action, reward, value, log_prob, done))

    def learn(self, clip_range: float = 0.2, epochs: int = 4):
        if not self.memory:
            return
        obs = torch.cat([t.obs for t in self.memory], dim=0)
        actions = torch.tensor([t.action for t in self.memory], device=self.device)
        rewards = [t.reward for t in self.memory]
        values = torch.cat([t.value for t in self.memory]).detach().squeeze(-1)
        log_probs = torch.cat([t.log_prob for t in self.memory]).detach()
        # Compute returns
        returns = []
        G = 0.0
        for r, t in zip(reversed(rewards), reversed(self.memory)):
            G = r + self.gamma * G * (1.0 - float(t.done))
            returns.insert(0, G)
        returns = torch.tensor(returns, device=self.device)
        advantages = returns - values
        for _ in range(epochs):
            new_log_probs, entropy, new_values = self.policy.evaluate_actions(obs, self.policy.init_hidden(obs.size(0), self.device), actions)
            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze(-1)).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "PPOLSTMAgent":
        device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a dummy observation and action space to rebuild the model
        dummy_obs = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, shape=(240, 160, 3), dtype=np.uint8),
            "party": gym.spaces.Box(0, 65535, shape=(6, 3), dtype=np.int32),
            "position": gym.spaces.Box(0, 65535, shape=(2,), dtype=np.int32),
        })
        dummy_act = gym.spaces.Discrete(len(cls.action_list))
        agent = cls(dummy_obs, dummy_act)
        agent.policy.load_state_dict(torch.load(path, map_location=device_obj))
        agent.policy.to(device_obj)
        agent.device = device_obj
        agent.reset_hidden()
        return agent
