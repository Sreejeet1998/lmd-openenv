"""
manual_test.py — smoke test for the LMD environment (no server needed).
Run from the lmd/ directory: python manual_test.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from server.lmd_environment import LmdEnvironment
from models import LmdAction, OrderStatus

def test_difficulty(difficulty: str, seed: int = 42):
    random.seed(seed)
    env = LmdEnvironment(difficulty=difficulty)
    obs = env.reset()

    print(f"\n=== {difficulty.upper()} ===")
    print(f"  Orders: {len(obs.orders)}, Vehicles: {len(obs.vehicles)}")

    steps = 0
    total_reward = 0.0
    max_steps = len(env._orders) + 10

    while not env._is_done() and steps < max_steps:
        pending  = [o for o in env._orders   if o.status == OrderStatus.PENDING]
        vehicles = [v for v in env._vehicles if not v.is_broken]
        if not pending or not vehicles:
            break

        import math
        target = pending[0]
        vehicle = min(vehicles, key=lambda v: math.dist(v.location, target.location))

        res = env.step(LmdAction(order_id=target.id, vehicle_id=vehicle.id))
        total_reward += res.reward
        steps += 1

    delivered = env._delivered_count
    total     = len(env._orders)
    score     = round(total_reward / max(total, 1), 4)
    print(f"  Steps: {steps}, Delivered: {delivered}/{total}, Score: {score}")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    assert delivered <= total,  f"Delivered > total: {delivered} > {total}"
    print(f"  PASS")
    return score


if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        test_difficulty(diff)
    print("\nAll manual tests passed.")
