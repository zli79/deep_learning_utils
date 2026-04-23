# Deep Learning Utils

A tutorial repository covering fundamental algorithms in deep learning and reinforcement learning, implemented from scratch in PyTorch.

## Contents

### Reinforcement Learning (`rl_alg/`)

| File | Description |
|------|-------------|
| `ppo.py` | Proximal Policy Optimization (PPO) — policy loss with clipping, value loss, and KL penalty |

### Parallel Training (`parallel_training/`)

| File | Description |
|------|-------------|
| `tensor_parallelism.py` | Column-wise tensor parallelism for `y = x @ W`, simulated with threads and message-passing queues |

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Running Examples

```bash
uv run rl_alg/ppo.py
uv run parallel_training/tensor_parallelism.py
```
