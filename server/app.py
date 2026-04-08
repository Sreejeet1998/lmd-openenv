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

@app.get("/health")
def health():
    return {"status": "ok"}

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
