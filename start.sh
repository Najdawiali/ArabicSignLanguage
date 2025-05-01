#!/usr/bin/env bash
# Start FastAPI app using uvicorn
uvicorn app:app --host 0.0.0.0 --port $PORT
