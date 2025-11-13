"""Vercel FastAPI entrypoint.

Exposes the ASGI application object named `app` for Vercel's Python serverless runtime.
The real application lives in `stellar_platform.serving.api`.

Vercel looks for one of several conventional filenames (app.py, main.py, api/app.py, etc.)
and an `app` variable. This thin wrapper satisfies that requirement.
"""
from stellar_platform.serving.api import app  # noqa: F401

# Simple lightweight liveness endpoint (distinct from the richer /health in the core API)
@app.get("/healthz")
async def healthz():  # pragma: no cover
	return {"status": "ok"}

# Optionally, you could customize startup/shutdown events here or wrap with middleware.
# Example:
# from fastapi import FastAPI
# base_app = app
# app = FastAPI()
# app.mount("/", base_app)
