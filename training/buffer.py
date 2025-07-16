from __future__ import annotations

from typing import Dict
import torch

class ReplayBuffer:
    """Small circular replay buffer for DVI-RL.

    Parameters
    ----------
    capacity: int
        Maximum number of entries stored.
    device: torch.device
        Device on which tensors are stored.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=4, device=torch.device('cpu'))
    >>> h = torch.randn(2)
    >>> buf.append(h, token=1, reward=1.0, conf=0.9)
    >>> len(buf)
    1
    >>> sample = buf.sample(1)
    >>> list(sample.keys())
    ['hidden', 'token', 'reward', 'conf']
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self._next_idx = 0
        self._size = 0
        self._hidden_buf: torch.Tensor | None = None
        self._token_buf: torch.Tensor | None = None
        self._reward_buf: torch.Tensor | None = None
        self._conf_buf: torch.Tensor | None = None

    def _allocate(self, hidden: torch.Tensor) -> None:
        shape = (self.capacity,) + hidden.shape
        self._hidden_buf = torch.empty(
            shape, dtype=hidden.dtype, device=self.device
        )
        self._token_buf = torch.empty(
            self.capacity, dtype=torch.long, device=self.device
        )
        self._reward_buf = torch.empty(
            self.capacity, dtype=torch.float32, device=self.device
        )
        self._conf_buf = torch.empty(
            self.capacity, dtype=torch.float32, device=self.device
        )

    def append(
        self,
        hidden: torch.Tensor,
        token: int,
        reward: float,
        conf: float,
    ) -> bool:
        if self._hidden_buf is None:
            self._allocate(hidden.detach())
        assert self._hidden_buf is not None
        dropped = self._size == self.capacity
        idx = self._next_idx
        self._hidden_buf[idx] = hidden.detach().to(self.device)
        self._token_buf[idx] = int(token)
        self._reward_buf[idx] = float(reward)
        self._conf_buf[idx] = float(conf)

        self._next_idx = (self._next_idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        return dropped

    def accepted_count(self) -> int:
        if self._size == 0:
            return 0
        return int((self._reward_buf[: self._size] == 1).sum().item())

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, accepted_only: bool = True) -> Dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("Buffer is empty")
        if accepted_only:
            mask = self._reward_buf[: self._size] == 1
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
        else:
            indices = torch.arange(self._size, device=self.device)
        if indices.numel() < batch_size:
            raise ValueError("Insufficient samples for requested batch size")
        perm = torch.randperm(indices.numel(), device=self.device)[:batch_size]
        sel = indices[perm]
        return {
            "hidden": self._hidden_buf[sel],
            "token": self._token_buf[sel],
            "reward": self._reward_buf[sel],
            "conf": self._conf_buf[sel],
        }

    def clear(self, accepted_only: bool = False) -> None:
        if self._size == 0:
            return
        if not accepted_only:
            self._next_idx = 0
            self._size = 0
            return
        mask = self._reward_buf[: self._size] != 1
        keep = mask.sum().item()
        if keep:
            self._hidden_buf[:keep] = self._hidden_buf[: self._size][mask]
            self._token_buf[:keep] = self._token_buf[: self._size][mask]
            self._reward_buf[:keep] = self._reward_buf[: self._size][mask]
            self._conf_buf[:keep] = self._conf_buf[: self._size][mask]
        self._size = int(keep)
        self._next_idx = self._size % self.capacity

    def to(self, device: torch.device) -> "ReplayBuffer":
        self.device = device
        if self._hidden_buf is not None:
            self._hidden_buf = self._hidden_buf.to(device)
            self._token_buf = self._token_buf.to(device)
            self._reward_buf = self._reward_buf.to(device)
            self._conf_buf = self._conf_buf.to(device)
        return self
