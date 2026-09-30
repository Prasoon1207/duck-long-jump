import torch


class RunningNormalizer:
    """Tracks a running mean/variance of observations and standardizes them."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = eps
        self.clip, self.eps = clip, eps

    def update(self, x: torch.Tensor) -> None:
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s: dict) -> None:
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]
