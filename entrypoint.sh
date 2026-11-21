#!/bin/bash
set -e

# Sanitize REDIS_URL in case code or extra quotes were pasted
if [[ -n "$REDIS_URL" ]]; then
  EXTRACTED=$(echo "$REDIS_URL" | grep -oE "(rediss?://[^[:space:]\"'\''\)]+)" | head -n 1 || true)
  if [[ -n "$EXTRACTED" ]]; then
    export REDIS_URL="$EXTRACTED"
  fi
fi

# Always start local Redis daemon in the background as a fail-safe fallback
echo "Starting local Redis daemon..."
redis-server --daemonize yes --dir /tmp --logfile /tmp/redis.log --pidfile /tmp/redis.pid || true

# Render injects $PORT (usually 10000); default to 10000
PORT="${PORT:-10000}"
echo "Starting FastAPI on 0.0.0.0:$PORT..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
