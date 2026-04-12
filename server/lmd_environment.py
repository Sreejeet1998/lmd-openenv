import random
import math
from uuid import uuid4
from typing import List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import LmdAction, LmdObservation, Order, Vehicle, OrderStatus
except (ImportError, ModuleNotFoundError):
    from server.models import LmdAction, LmdObservation, Order, Vehicle, OrderStatus


class LmdEnvironment(Environment):
    """
    A Last Minute Delivery (LMD) simulation environment.

    Tasks:
      easy   — 5 orders, 1 vehicle, no constraints
      medium — 10 orders, 2 vehicles, weight capacity
      hard   — 15 orders, 3 vehicles, time windows + priority + vehicle breakdown
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        self._difficulty = difficulty
        self._seed = seed
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.reset()

    def _generate_orders(self, count: int, use_weights: bool = False, use_time: bool = False) -> List[Order]:
        orders = []
        for i in range(count):
            loc = (random.uniform(0, 10), random.uniform(0, 10))
            weight = random.uniform(1, 20) if use_weights else 5.0
            if use_time:
                start = random.uniform(0, 10)
                end = start + random.uniform(5, 15)
                time_window = (start, end)
            else:
                time_window = (0.0, 1000.0)
            orders.append(Order(
                id=f"order_{i}",
                location=loc,
                weight=weight,
                time_window=time_window,
                status=OrderStatus.PENDING,
                priority=random.randint(1, 3) if use_time else 1,
            ))
        return orders

    def _generate_vehicles(self, count: int) -> List[Vehicle]:
        return [
            Vehicle(
                id=f"vehicle_{i}",
                location=(5.0, 5.0),
                capacity=100.0,
                max_capacity=100.0,
            )
            for i in range(count)
        ]

    def reset(self, difficulty: str = None) -> LmdObservation:
        if difficulty:
            self._difficulty = difficulty
        if self._seed is not None:
            random.seed(self._seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_time = 0.0
        self._total_distance = 0.0
        self._delivered_count = 0
        self._failed_count = 0
        self._capacity_violations = 0
        self._time_violations = 0
        self._weather = random.choice(["sunny", "rainy", "stormy"]) if self._difficulty in ["medium", "hard"] else "sunny"
        self._breakdown_triggered = False
        self._last_reward = 0.0

        if self._difficulty == "easy":
            self._orders = self._generate_orders(5)
            self._vehicles = self._generate_vehicles(1)
        elif self._difficulty == "medium":
            self._orders = self._generate_orders(10, use_weights=True)
            self._vehicles = self._generate_vehicles(2)
        elif self._difficulty == "hard":
            self._orders = self._generate_orders(15, use_weights=True, use_time=True)
            self._vehicles = self._generate_vehicles(3)
        else:
            raise ValueError(f"Unknown difficulty: {self._difficulty!r}")

        return self._make_observation("Environment reset.")

    def _make_observation(self, message: str = "") -> LmdObservation:
        # Traffic varies by time of day (sine wave + small noise)
        traffic = 1.0 + 0.5 * math.sin(self._current_time * math.pi / 20.0) 
        traffic = max(0.5, traffic) # Ensure non-zero
        
        # Add visual map to all messages
        display_msg = message + self._render_ascii_map()
        
        return LmdObservation(
            orders=self._orders,
            vehicles=self._vehicles,
            current_time=self._current_time,
            task_difficulty=self._difficulty,
            weather=self._weather,
            traffic_level=round(traffic, 2),
            message=display_msg,
            done=self._is_done(),
            reward=self._last_reward,
        )

    def _is_done(self) -> bool:
        all_assigned = all(o.status != OrderStatus.PENDING for o in self._orders)
        if all_assigned:
            return True
        # Deadlock: pending orders but no working vehicle
        if not any(not v.is_broken for v in self._vehicles):
            for o in self._orders:
                if o.status == OrderStatus.PENDING:
                    o.status = OrderStatus.FAILED
                    self._failed_count += 1
            return True
        # Energy failure: all active vehicles out of battery
        if all(v.battery_level <= 0 for v in self._vehicles if not v.is_broken):
            return True
        return False

    def _render_ascii_map(self) -> str:
        grid = [["." for _ in range(11)] for _ in range(11)]
        # Mark orders
        for o in self._orders:
            if o.status == OrderStatus.PENDING:
                r, c = int(o.location[1]), int(o.location[0])
                grid[r][c] = "O"
        # Mark vehicles
        for v in self._vehicles:
            r, c = int(v.location[1]), int(v.location[0])
            grid[r][c] = "V" if not v.is_broken else "X"
        
        map_str = "\n".join(" ".join(row) for row in reversed(grid))
        return f"\nCity Grid (V:Vehicle, O:Order, X:Broken):\n{map_str}\n"

    def _calculate_reward(self, in_time: bool, has_capacity: bool, distance: float, priority: int = 1) -> float:
        reward = 0.30  # base delivery success
        if has_capacity:
            reward += 0.20
        if in_time:
            reward += 0.20
        
        # Distance efficiency (capped)
        dist_score = max(0.0, 0.30 * (1.0 - distance / 15.0))
        reward += dist_score
        
        # Weather penalty adjustment
        if self._weather == "rainy":
            reward *= 0.95
        elif self._weather == "stormy":
            reward *= 0.90

        # Priority bonus on hard mode
        if self._difficulty == "hard" and priority > 1:
            reward *= 1.0 + 0.1 * (priority - 1)
            
        return round(min(max(reward, 0.0), 1.0), 4)

    def step(self, action: LmdAction) -> LmdObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._difficulty == "hard" and not self._breakdown_triggered and self._state.step_count >= 5:
            self._vehicles[0].is_broken = True
            self._breakdown_triggered = True

        if action.replan:
            return self._make_observation("Replanned.")

        if not action.order_id or not action.vehicle_id:
            return self._make_observation("Missing order_id or vehicle_id.")

        order   = next((o for o in self._orders   if o.id == action.order_id),   None)
        vehicle = next((v for v in self._vehicles if v.id == action.vehicle_id), None)

        if not order:
            return self._make_observation(f"Order {action.order_id!r} not found.")
        if not vehicle:
            return self._make_observation(f"Vehicle {action.vehicle_id!r} not found.")
        if vehicle.is_broken:
            return self._make_observation(f"Vehicle {vehicle.id} is broken.")
        if order.status != OrderStatus.PENDING:
            return self._make_observation(f"Order {order.id} already processed.")

        order.status = OrderStatus.ASSIGNED

        # Update environment state
        dist = math.sqrt(
            (vehicle.location[0] - order.location[0]) ** 2 +
            (vehicle.location[1] - order.location[1]) ** 2
        )
        
        # Calculate effective travel time with weather and traffic
        weather_factor = {"sunny": 1.0, "rainy": 1.3, "stormy": 1.8}[self._weather]
        traffic_factor = 1.0 + 0.5 * math.sin(self._current_time * math.pi / 20.0)
        effective_dist = dist * weather_factor * max(1.0, traffic_factor)

        self._total_distance += dist
        self._current_time   += effective_dist
        
        # Energy and Capacity
        drain = dist * 2.0  # 2% per unit distance
        vehicle.battery_level = max(0.0, vehicle.battery_level - drain)
        
        has_capacity = vehicle.capacity >= order.weight
        if not has_capacity:
            self._capacity_violations += 1
        else:
            vehicle.capacity -= order.weight
            
        in_time = order.time_window[0] <= self._current_time <= order.time_window[1]
        if not in_time:
            self._time_violations += 1

        # Finalize step
        msg = f"Delivered {order.id} via {vehicle.id}."
        if vehicle.battery_level <= 0:
            vehicle.is_broken = True
            msg += f" [WARNING: {vehicle.id} out of battery!]"

        vehicle.location = order.location
        order.status      = OrderStatus.DELIVERED
        self._delivered_count += 1

        self._last_reward = self._calculate_reward(in_time, has_capacity, dist, order.priority)
        return self._make_observation(msg)

    @property
    def state(self) -> State:
        return self._state
