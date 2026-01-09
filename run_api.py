#!/usr/bin/env python
"""Simple script to run the Stellar Platform API server."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stellar_platform.serving.api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info",
    )
