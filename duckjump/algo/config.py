from dataclasses import dataclass


@dataclass
class PPOConfig:
    # data collection
    num_envs: int = 8
    rollout_steps: int = 256          # per env, so one update sees num_envs * rollout_steps samples
    total_steps: int = 5_000_000

    # returns / advantages
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # optimisation
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 5
    num_minibatches: int = 8
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: float | None = 0.03    # stop the epoch loop early if the policy moves too far
    normalize_advantages: bool = True
    normalize_observations: bool = True

    # networks
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    init_log_std: float = -0.7

    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches
