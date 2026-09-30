"""On-policy rollout storage and Generalized Advantage Estimation (GAE)."""

import torch


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, act_dim: int):
        self.num_steps, self.num_envs = num_steps, num_envs
        self.obs = torch.zeros(num_steps, num_envs, obs_dim)
        self.actions = torch.zeros(num_steps, num_envs, act_dim)
        self.log_probs = torch.zeros(num_steps, num_envs)
        self.rewards = torch.zeros(num_steps, num_envs)
        self.dones = torch.zeros(num_steps, num_envs)   # 1 if the episode ended at this step
        self.values = torch.zeros(num_steps, num_envs)
        self.advantages = torch.zeros(num_steps, num_envs)
        self.returns = torch.zeros(num_steps, num_envs)
        self.t = 0

    def add(self, obs, actions, log_probs, rewards, dones, values):
        t = self.t
        self.obs[t], self.actions[t], self.log_probs[t] = obs, actions, log_probs
        self.rewards[t], self.dones[t], self.values[t] = rewards, dones, values
        self.t += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """A_t = sum_k (gamma*lam)^k * delta_{t+k},  delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)."""
        gae = torch.zeros(self.num_envs)
        for t in reversed(range(self.num_steps)):
            next_value = last_value if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def minibatches(self, num_minibatches: int, generator: torch.Generator):
        n = self.num_steps * self.num_envs
        flat = {
            "obs": self.obs.reshape(n, -1),
            "actions": self.actions.reshape(n, -1),
            "log_probs": self.log_probs.reshape(n),
            "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }
        perm = torch.randperm(n, generator=generator)
        for idx in perm.chunk(num_minibatches):
            yield {k: v[idx] for k, v in flat.items()}

    def clear(self):
        self.t = 0
