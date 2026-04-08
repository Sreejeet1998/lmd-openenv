---
title: Last Mile Delivery Environment
emoji: 🚛
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Last Mile Delivery (LMD) Environment

A simulation environment where an AI agent manages a fleet of delivery vehicles to fulfill customer orders across a synthetic city grid. 

## Motivation
Logistics and last-mile delivery represent one of the most complex optimization problems in the modern economy. Inefficiencies lead to higher costs, increased carbon emissions, and poorer customer service. This environment provides a playground for developing AI agents that can navigate complex real-world constraints like vehicle capacity, time windows, and unexpected equipment failures to optimize delivery operations.

## Tasks & Difficulty

| Level  | Orders | Vehicles | Constraints |
|--------|--------|----------|-------------|
| easy   | 5      | 1        | None — basic routing |
| medium | 10     | 2        | Weight capacity constraints |
| hard   | 15     | 3        | Time windows + priorities + vehicle breakdown |

## Environment Variables (Required for Inference)

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
```

## Quick Start

```bash
# Install dependencies
python -m pip install -e .

# Run inference script (root of repo)
python inference.py

# Run baseline agent
python -m server.baseline_agent

# Run smoke tests
python -m server.manual_test
```

## Inference Script

`inference.py` is at the **root** of the repository and emits structured JSON logs strictly following the `[START]`, `[STEP]`, and `[END]` format.

## Reward System

Per-step reward (0.0–1.0):

| Component | Weight | Condition |
|-----------|--------|-----------|
| Delivery success | 0.30 | Order reached |
| Capacity respected | 0.20 | Vehicle not overloaded |
| Time window met | 0.20 | Arrival within window |
| Distance efficiency | 0.30 | Less travel = more reward |

Hard mode adds a priority multiplier (×1.0/×1.1/×1.2) for priority 1/2/3 orders.

## Action Space

**LmdAction**:
- `order_id` (str): ID of the order to assign
- `vehicle_id` (str): ID of the vehicle to use
- `replan` (bool): Trigger a replan signal

## Observation Space

**LmdObservation**:
- `orders` (List[Order]): All orders with location, weight, time_window, status, priority
- `vehicles` (List[Vehicle]): All vehicles with location, capacity, is_broken
- `current_time` (float): Elapsed simulation time
- `task_difficulty` (str): Current task level
- `message` (str): Step result message
- `reward` (float): Last step reward
- `done` (bool): Episode completion status

## Graders

Combined score formula (0.0–1.0):

```
score = 0.40 × delivery_rate + 0.20 × capacity_score + 0.20 × time_score + 0.20 × reward_score
```

## Baseline Scores (seed=42)

| Task   | Score |
|--------|-------|
| easy   | 0.9791 |
| medium | 0.9873 |
| hard   | 0.9658 |

## Project Structure

```
.
├── Dockerfile                 ← Containerization config
├── inference.py               ← Mandatory inference entry point
├── openenv.yaml               ← OpenEnv specification
├── pyproject.toml             ← Python metadata & dependencies
├── uv.lock                    ← Reproducible lockfile
└── server/                    ← Environment package
    ├── app.py                 ← FastAPI server & /health endpoint
    ├── lmd_environment.py     ← Core simulation logic
    ├── models.py              ← Pydantic data models
    ├── baseline_agent.py      ← Greedy baseline implementation
    ├── client.py              ← Environment client
    └── manual_test.py         ← Local smoke tests
```
