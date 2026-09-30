"""Proximal Policy Optimization (clipped objective) with an actor-critic network."""

import numpy as np
import torch
from torch import nn

from duckjump.algo.buffer import RolloutBuffer
from duckjump.algo.config import PPOConfig
from duckjump.algo.networks import ActorCritic
from duckjump.algo.normalizer import RunningNormalizer


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.net = ActorCritic(obs_dim, act_dim, cfg.actor_hidden, cfg.critic_hidden, cfg.init_log_std)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.obs_norm = RunningNormalizer(obs_dim) if cfg.normalize_observations else None
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, obs_dim, act_dim)

    # ---------------------------------------------------------------- acting

    def _prep(self, obs: np.ndarray, update_stats: bool = False) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32)
        if self.obs_norm is None:
            return x
        if update_stats:
            self.obs_norm.update(x)
        return self.obs_norm(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Returns (action, log_prob, value, normalized_obs) for a batch of observations."""
        x = self._prep(obs, update_stats=not deterministic)
        action, log_prob, value = self.net.act(x, deterministic)
        return action, log_prob, value, x

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> torch.Tensor:
        return self.net.value(self._prep(obs))

    def store(self, norm_obs, action, log_prob, value, reward, done):
        self.buffer.add(norm_obs, action, log_prob, torch.as_tensor(reward, dtype=torch.float32),
                        torch.as_tensor(done, dtype=torch.float32), value)

    def bootstrap_truncated(self, reward: np.ndarray, truncated: np.ndarray, final_obs: np.ndarray) -> np.ndarray:
        """A time-limit cut is not a real ending: credit the value of the state we were cut off in."""
        reward = reward.copy()
        if truncated.any():
            reward[truncated] += self.cfg.gamma * self.value(final_obs[truncated]).numpy()
        return reward

    # -------------------------------------------------------------- learning

    def set_progress(self, fraction_done: float) -> None:
        if self.cfg.anneal_lr:
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.learning_rate * max(1.0 - fraction_done, 0.05)

    def update(self, last_obs: np.ndarray) -> dict:
        cfg, buf = self.cfg, self.buffer
        buf.compute_gae(self.value(last_obs), cfg.gamma, cfg.gae_lambda)

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(cfg.update_epochs):
            epoch_kl = []
            for mb in buf.minibatches(cfg.num_minibatches, self.generator):
                new_logp, entropy, new_value = self.net.evaluate(mb["obs"], mb["actions"])
                log_ratio = new_logp - mb["log_probs"]
                ratio = log_ratio.exp()

                adv = mb["advantages"]
                if cfg.normalize_advantages:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped surrogate: don't let the new policy profit from moving too far.
                surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv)
                policy_loss = -surrogate.mean()

                if cfg.clip_value_loss:
                    v_clipped = mb["values"] + (new_value - mb["values"]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    value_loss = 0.5 * torch.max((new_value - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()

                entropy_bonus = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
                epoch_kl.append(approx_kl)
                for k, v in zip(stats, [policy_loss.item(), value_loss.item(), entropy_bonus.item(), approx_kl, clip_frac]):
                    stats[k].append(v)
            if cfg.target_kl is not None and np.mean(epoch_kl) > cfg.target_kl:
                break

        buf.clear()
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["action_std"] = self.net.log_std.exp().mean().item()
        out["explained_variance"] = _explained_variance(buf.values.flatten(), buf.returns.flatten())
        return out

    # ------------------------------------------------------------ checkpoints

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_norm": self.obs_norm.state_dict() if self.obs_norm else None,
        }

    def load_state_dict(self, s: dict) -> None:
        self.net.load_state_dict(s["net"])
        self.optimizer.load_state_dict(s["optimizer"])
        if self.obs_norm and s["obs_norm"]:
            self.obs_norm.load_state_dict(s["obs_norm"])


def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    return float("nan") if var == 0 else float(1 - (target - pred).var() / var)
