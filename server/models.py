# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Lmd Environment.

The lmd environment is a simple test environment that echoes back messages.
"""

from enum import Enum
from typing import List, Optional, Tuple
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    DELIVERED = "delivered"
    FAILED = "failed"


class Order(BaseModel):
    id: str
    location: Tuple[float, float]
    weight: float
    time_window: Tuple[float, float]
    status: OrderStatus = OrderStatus.PENDING
    priority: int = 1


class Vehicle(BaseModel):
    id: str
    location: Tuple[float, float]
    capacity: float
    max_capacity: float
    is_broken: bool = False


class LmdAction(Action):
    """Action for the Lmd environment."""
    order_id: Optional[str] = Field(None, description="ID of the order to assign")
    vehicle_id: Optional[str] = Field(None, description="ID of the vehicle to assign to")
    replan: bool = Field(False, description="Whether to trigger a replan action")


class LmdObservation(Observation):
    """Observation from the Lmd environment."""
    orders: List[Order] = Field(default_factory=list)
    vehicles: List[Vehicle] = Field(default_factory=list)
    current_time: float = 0.0
    task_difficulty: str = "easy"
    message: str = Field(default="", description="Status message")
    reward: float = 0.0
    done: bool = False
