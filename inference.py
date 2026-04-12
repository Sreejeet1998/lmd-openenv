"""
inference.py — LMD Hackathon Inference Script
Root-level entry point. Implements mandatory [START]/[STEP]/[END] log format.
Follows precisely the Sample Inference Script structure and naming.
"""

import os
import sys
import json
import time
import math
import random
from typing import List, Optional, Dict, Any
from openai import OpenAI

# ── Environment Configuration ────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

# ── Local Imports ───────────────────────────────────────────────────
from server.lmd_environment import LmdEnvironment
from server.models import LmdAction, LmdObservation, OrderStatus

# ── OpenAI Client ───────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "hf-no-key",
    timeout=20.0,
)

# ── Constants ───────────────────────────────────────────────────────
SUCCESS_SCORE_THRESHOLD = 0.75
GRADER_SEEDS = [42, 123, 7, 99, 256]

# ── Logging Helpers ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    # env should match name in openenv.yaml
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str] = None):
    # Using compact JSON for the action dictionary
    action_str = json.dumps(action, separators=(',', ':'))
    # Ensure all values are correctly formatted
    log_msg = f"[STEP] step={step} action={action_str} reward={float(round(reward, 4))} done={str(done).lower()}"
    if error:
        log_msg += f" error={json.dumps(error)}"
    print(log_msg, flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]):
    # Ensure success is lowercase 'true'/'false'
    print(f"[END] task={task} score={float(round(score, 4))} steps={steps} success={str(success).lower()}", flush=True)

# ── Agent Heuristics & LLM calls ────────────────────────────────────

def _build_prompt(obs: LmdObservation, difficulty: str) -> str:
    pending  = [o for o in obs.orders   if o.status == OrderStatus.PENDING]
    vehicles = [v for v in obs.vehicles if not v.is_broken]

    order_lines = "\n".join(
        f"  - {o.id}: loc=({o.location[0]:.1f},{o.location[1]:.1f}) "
        f"weight={o.weight:.1f} priority={o.priority} "
        f"window=[{o.time_window[0]:.1f},{o.time_window[1]:.1f}]"
        for o in pending
    )
    vehicle_lines = "\n".join(
        f"  - {v.id}: loc=({v.location[0]:.1f},{v.location[1]:.1f}) "
        f"capacity={v.capacity:.1f}/{v.max_capacity:.1f}"
        for v in vehicles
    )

    return f"""You are a last-mile delivery dispatcher. Choose ONE delivery to make next.

Difficulty: {difficulty}
Current time: {obs.current_time:.1f}
Weather: {obs.weather}
Traffic: {obs.traffic_level}x slower

PENDING ORDERS:
{order_lines if order_lines else '  (none)'}

AVAILABLE VEHICLES:
{vehicle_lines if vehicle_lines else '  (none — all broken)'}

Respond with valid JSON only:
{{"order_id": "<order_id>", "vehicle_id": "<vehicle_id>"}}

Rules:
- Primary: Pick the order with the earliest time-window deadline.
- Secondary: If multiple deadlines are far off, pick the nearest one.
- Logistics: Account for weather/traffic delays. Assign to vehicles with high battery/capacity.
- If no orders or vehicles remain, respond: {{"order_id": null, "vehicle_id": null}}
"""

def _call_llm(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a delivery dispatcher. Respond in valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)
    except Exception:
        return {"order_id": None, "vehicle_id": None}

def _greedy_fallback(obs: LmdObservation) -> dict:
    pending  = [o for o in obs.orders   if o.status == OrderStatus.PENDING]
    vehicles = [v for v in obs.vehicles if not v.is_broken]
    if not pending or not vehicles:
        return {"order_id": None, "vehicle_id": None}
    target_order   = min(pending,   key=lambda o: o.time_window[1])
    target_vehicle = min(vehicles,  key=lambda v: math.dist(v.location, target_order.location))
    return {"order_id": target_order.id, "vehicle_id": target_vehicle.id}

# ── Inference Loop ──────────────────────────────────────────────────

def run_episode(difficulty: str, seed: int, track_logs: bool = False):
    random.seed(seed)
    env = LmdEnvironment(difficulty=difficulty)
    obs = env.reset()
    
    rewards = []
    steps_taken = 0
    max_steps = len(env._orders) + 5
    
    task_id = difficulty  # Matches tasks[].id in openenv.yaml
    if track_logs:
        log_start(task=task_id, env="lmd", model=MODEL_NAME)

    for step in range(1, max_steps + 1):
        if env._is_done():
            break
            
        decision = None
        if HF_TOKEN and HF_TOKEN != "hf-no-key":
            decision = _call_llm(_build_prompt(obs, difficulty))
        
        if not decision or not decision.get("order_id") or not decision.get("vehicle_id"):
            decision = _greedy_fallback(obs)
            
        order_id = decision.get("order_id")
        vehicle_id = decision.get("vehicle_id")
            
        if not order_id or not vehicle_id:
            break
            
        action = LmdAction(order_id=order_id, vehicle_id=vehicle_id)
        obs = env.step(action)
        reward = obs.reward
        done = obs.done
        
        rewards.append(reward)
        steps_taken = step
        
        if track_logs:
            log_step(step=step, action=action.model_dump(), reward=reward, done=done)
            
        if done:
            break
            
    # Calculate weighted score for grading
    delivery_rate = env._delivered_count / max(len(env._orders), 1)
    capacity_score = max(0.0, 1.0 - (env._capacity_violations / max(len(env._orders), 1)))
    time_score = max(0.0, 1.0 - (env._time_violations / max(len(env._orders), 1)))
    avg_reward = sum(rewards) / max(len(env._orders), 1)
    
    score = (0.40 * delivery_rate + 0.20 * capacity_score + 0.20 * time_score + 0.20 * avg_reward)
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    
    if track_logs:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)
        
    return score

def main():
    # If TASK_ID is provided by the validator, run only that task
    target_task = os.environ.get("TASK_ID")
    
    if target_task:
        # Clean up task ID (handle both 'easy' and 'lmd_easy')
        clean_task = target_task.replace("lmd_", "").lower()
        if clean_task in ["easy", "medium", "hard"]:
            run_episode(clean_task, 42, track_logs=True)
        else:
            print(f"Unknown task: {target_task}. Expected one of: easy, medium, hard", file=sys.stderr)
    else:
        # Local testing: run all tasks
        for diff in ["easy", "medium", "hard"]:
            run_episode(diff, 42, track_logs=True)
            
            # Optional: calculate average cross-seed score for internal verification
            seed_scores = [run_episode(diff, s, track_logs=False) for s in GRADER_SEEDS]
            mean_score = sum(seed_scores) / len(seed_scores)
            print(f"[INTERNAL] Task {diff} Mean Score: {mean_score:.4f}", file=sys.stderr)

if __name__ == "__main__":
    main()