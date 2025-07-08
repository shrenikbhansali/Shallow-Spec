import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import pytest

from training.buffer import ReplayBuffer


def test_append_wraparound():
    buf = ReplayBuffer(capacity=3, device=torch.device('cpu'))
    for i in range(4):
        buf.append(torch.full((2,), float(i)), token=i, reward=float(i % 2), conf=0.0)
    assert len(buf) == 3
    tokens = buf._token_buf[: len(buf)].tolist()
    assert sorted(tokens) == [1, 2, 3]
    assert buf.accepted_count() == 2


def test_accepted_count():
    buf = ReplayBuffer(capacity=4, device=torch.device('cpu'))
    rewards = [1.0, 0.0, 1.0]
    for i, r in enumerate(rewards):
        buf.append(torch.zeros(1), token=i, reward=r, conf=0.0)
    assert buf.accepted_count() == 2


def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=5, device=torch.device('cpu'))
    for i in range(5):
        buf.append(torch.zeros(4), token=i, reward=1.0, conf=0.5)
    batch = buf.sample(3)
    assert batch['hidden'].shape == (3, 4)
    assert batch['token'].dtype == torch.long
    assert batch['reward'].dtype == torch.float32
    assert batch['conf'].dtype == torch.float32


def test_clear_all():
    buf = ReplayBuffer(capacity=3, device=torch.device('cpu'))
    for i in range(3):
        buf.append(torch.zeros(1), token=i, reward=1.0, conf=0.0)
    buf.clear()
    assert len(buf) == 0
    assert buf.accepted_count() == 0


def test_clear_accepted_only():
    buf = ReplayBuffer(capacity=5, device=torch.device('cpu'))
    rewards = [1.0, 0.0, 1.0, 0.0]
    for i, r in enumerate(rewards):
        buf.append(torch.zeros(1), token=i, reward=r, conf=0.0)
    buf.clear(accepted_only=True)
    assert buf.accepted_count() == 0
    tokens = buf._token_buf[: len(buf)].tolist()
    assert tokens == [1, 3]
