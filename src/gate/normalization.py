import numpy as np


class RunningStandardizer:
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        self.hidden_dim = hidden_dim
        self.eps = float(eps)

        self.count = 0

        self.mean = np.zeros(
            self.hidden_dim,
            dtype=np.float64,
        )

        self.M2 = np.zeros(
            self.hidden_dim,
            dtype=np.float64,
        )

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"expected [N, H], got {X.shape}")

        if X.shape[1] != self.hidden_dim:
            raise ValueError(f"Expected H = {self.hidden_dim} got {X.shape[1]}")

        if X.shape[0] == 0:
            return
        n = X.shape[0]

        batch_mean = X.mean(axis=0)

        centered = X - batch_mean

        batch_M2 = (centered * centered).sum(axis=0)

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.M2 = batch_M2
            return

        old_count = self.count
        new_count = self.count + n

        delta = batch_mean - self.mean

        self.mean = self.mean + delta * (n / new_count)
        self.M2 = self.M2 + batch_M2 + delta**2 * (old_count * n / new_count)

        self.count = new_count

    def std(self) -> np.ndarray:
        if self.count < 2:
            raise RuntimeError("Need at least 2 states.")

        var = self.M2 / self.count
        return np.sqrt(np.maximum(var, self.eps**2))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.count < 2:
            raise RuntimeError("Running statistics not initialized")

        X = np.asarray(X, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        std = self.std()

        return ((X - mean) / std).astype(np.float32)
