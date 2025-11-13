"""Alternative FastAPI entrypoint for Vercel.
Exports `app` so Vercel can detect the ASGI application (duplicate of app.py wrapper).
"""
from stellar_platform.serving.api import app  # noqa: F401
