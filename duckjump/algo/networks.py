"""Actor-critic: a Gaussian policy (actor) and a state-value function (critic)."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [_init(nn.Linear(d, h), np.sqrt(2)), nn.ELU()]
        d = h
    layers.append(_init(nn.Linear(d, out_dim), out_gain))
    return nn.Sequential(*layers)


def _init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, actor_hidden, critic_hidden, init_log_std: float):
        super().__init__()
        self.actor_mean = mlp(obs_dim, actor_hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.critic = mlp(obs_dim, critic_hidden, 1, out_gain=1.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor_mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action; returns (action, log_prob, value)."""
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Re-score stored actions under the current policy; returns (log_prob, entropy, value)."""
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)
