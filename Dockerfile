FROM python:3.11-slim

# Install system dependencies (ffmpeg is essential for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set up user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMOTION_MODEL_PATH=/home/user/app/models/emotion_cnn.pt

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user . .

# Ensure working directories exist and entrypoint is executable
RUN mkdir -p models/cache data/raw data/processed && chmod +x entrypoint.sh

EXPOSE 8000 10000

CMD ["./entrypoint.sh"]
