#!/bin/bash
set -e

# Sanitize REDIS_URL in case code or extra quotes were pasted
if [[ -n "$REDIS_URL" ]]; then
  EXTRACTED=$(echo "$REDIS_URL" | grep -oE "(rediss?://[^[:space:]\"'\''\)]+)" | head -n 1 || true)
  if [[ -n "$EXTRACTED" ]]; then
    export REDIS_URL="$EXTRACTED"
  fi
fi

# If using local Redis (default), start the background Redis daemon
if [[ -z "$REDIS_URL" || "$REDIS_URL" == *"localhost"* || "$REDIS_URL" == *"127.0.0.1"* ]]; then
  echo "Starting local Redis daemon..."
  redis-server --daemonize yes --dir /tmp --logfile /tmp/redis.log --pidfile /tmp/redis.pid
  until redis-cli ping > /dev/null 2>&1; do
    sleep 0.5
  done
  echo "Local Redis daemon is ready."
else
  echo "Using remote Redis at $REDIS_URL"
fi

# Render injects $PORT (usually 10000); default to 8000 for local runs
PORT="${PORT:-8000}"
echo "Starting FastAPI on 0.0.0.0:$PORT..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
