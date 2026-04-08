"""
baseline_agent.py — Greedy baseline for the LMD environment.

Run from the repo root:
    python -m lmd.baseline_agent          # or
    cd lmd && python baseline_agent.py
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from server.lmd_environment import LmdEnvironment
from models import LmdAction, OrderStatus


def solve_greedy(difficulty: str = "easy", seed: int = 42) -> float:
    """
    Greedy nearest-vehicle heuristic.
    Seeded for reproducible scores.
    Returns a score in [0.0, 1.0].
    """
    random.seed(seed)
    print(f"\n--- Running Baseline Agent (difficulty={difficulty}, seed={seed}) ---")

    env = LmdEnvironment(difficulty=difficulty)
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    while not env._is_done():
        pending  = [o for o in env._orders   if o.status == OrderStatus.PENDING]
        vehicles = [v for v in env._vehicles if not v.is_broken]

        if not pending or not vehicles:
            if not vehicles:
                print("  ALERT: All vehicles broken — episode terminated.")
            break

        # Pick order with earliest deadline (helps on hard mode)
        target_order = min(pending, key=lambda o: o.time_window[1])

        # Nearest working vehicle
        target_vehicle = min(
            vehicles,
            key=lambda v: math.sqrt(
                (v.location[0] - target_order.location[0]) ** 2 +
                (v.location[1] - target_order.location[1]) ** 2
            ),
        )

        res = env.step(LmdAction(order_id=target_order.id, vehicle_id=target_vehicle.id))
        total_reward += res.reward
        steps += 1

    num_orders  = len(env._orders)
    final_score = round(total_reward / max(num_orders, 1), 4)

    print(f"  Orders: {num_orders}  Delivered: {env._delivered_count}  Failed: {env._failed_count}")
    print(f"  Total reward: {total_reward:.3f}  Score: {final_score:.4f}")
    return final_score


if __name__ == "__main__":
    results = {}
    for diff in ["easy", "medium", "hard"]:
        results[diff] = solve_greedy(diff)

    print("\n" + "=" * 40)
    print("BASELINE EVALUATION RESULTS (seed=42)")
    print("=" * 40)
    for diff, score in results.items():
        bar = "#" * int(score * 20)
        print(f"  {diff:8s}  {score:.4f}  [{bar:<20}]")
    print("=" * 40)
