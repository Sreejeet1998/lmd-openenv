"""FastAPI application for the LMD Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openenv.core.env_server.http_server import create_app

try:
    from .models import LmdAction, LmdObservation
    from .lmd_environment import LmdEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from models import LmdAction, LmdObservation
        from lmd_environment import LmdEnvironment
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models import LmdAction, LmdObservation
        from lmd_environment import LmdEnvironment

from fastapi import FastAPI

app = create_app(
    LmdEnvironment,
    LmdAction,
    LmdObservation,
    env_name="lmd",
    max_concurrent_envs=1,
)

# Enhance API Metadata
app.title = "Last Mile Delivery (LMD) Environment"
app.description = """
### Logistics Simulation for Agentic AI

This environment simulates a real-world last-mile delivery scenario where an AI agent dispatches 
a fleet of vehicles to fulfill customer orders across a synthetic city grid.

**Key Features:**
* **Physics-based travel**: Travel time and reward are derived from Euclidean distances.
* **Constraints**: Agents must manage vehicle load capacity and delivery time windows.
* **Dynamic Challenges**: Hard mode introduces random vehicle breakdowns and order priorities.
* **Standardized Rewards**: All signals are normalized between 0.0 and 1.0 for cross-evaluation.

**Getting Started:**
1. Call `/reset` with a difficulty level to start an episode.
2. Call `/step` repeatedly to dispatch orders until `done` is true.
3. Use `/state` to track cumulative progress.
"""
app.version = "1.0.0"


@app.get("/health", summary="Health Check", description="Checks if the environment server is responsive.")
def health():
    return {"status": "ok"}


@app.get("/", summary="Root Redirect", description="Redirects to this interactive documentation page.")
def index():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
