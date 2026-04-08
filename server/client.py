# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lmd Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LmdAction, LmdObservation


class LmdEnv(
    EnvClient[LmdAction, LmdObservation, State]
):
    """
    Client for the Lmd Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with LmdEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     result = client.step(LmdAction(order_id="order_0", vehicle_id="vehicle_0"))
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LmdEnv.from_docker_image("lmd-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(LmdAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LmdAction) -> Dict:
        """
        Convert LmdAction to JSON payload for step message.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[LmdObservation]:
        """
        Parse server response into StepResult[LmdObservation].
        """
        observation = LmdObservation.model_validate(payload.get("observation", {}))
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
