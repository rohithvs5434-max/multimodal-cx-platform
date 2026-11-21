import asyncio
import contextlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

from app.config import Settings
from app.inference import MultimodalAnalyzer
from app.fusion import fuse
from app.resolution import build_attention, build_followup
from app.schemas import (
    CXEvent,
    DemoSeedResponse,
    EmotionResult,
    FollowUp,
    FusionResult,
    HealthResponse,
    SentimentResult,
)
from app.streams import EventStream


settings = Settings()
HANDLED_KEY = "cx:handled"
FOLLOWUP_PREFIX = "cx:followup:"
FOLLOWUP_TTL_SECONDS = 24 * 3600

app = FastAPI(title="Multimodal CX Platform", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup() -> None:
    redis = None
    if settings.redis_url:
        try:
            r = Redis.from_url(settings.redis_url, decode_responses=True)
            await asyncio.wait_for(r.ping(), timeout=3.0)
            redis = r
            print("Connected to configured Redis.")
        except Exception as exc:
            print(f"Warning: Could not connect to configured Redis ({exc})")

    if redis is None:
        try:
            local_r = Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
            await asyncio.wait_for(local_r.ping(), timeout=2.0)
            redis = local_r
            print("Connected to local fallback Redis.")
        except Exception as exc:
            print(f"Warning: Local fallback Redis also unavailable: {exc}")

    if redis is None:
        redis = Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)

    app.state.redis = redis
    app.state.events = EventStream(redis, settings.redis_stream_key, settings.event_retention)
    app.state.analyzer = MultimodalAnalyzer(settings)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.redis.aclose()


def requires_state() -> tuple[EventStream, MultimodalAnalyzer]:
    return app.state.events, app.state.analyzer


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    events, analyzer = requires_state()
    try:
        await events.client.ping()
    except Exception:
        return HealthResponse(status="degraded", redis=False, emotion_model_ready=analyzer.emotion_model_ready)
    return HealthResponse(status="ok", redis=True, emotion_model_ready=analyzer.emotion_model_ready)


@app.get("/api/events", response_model=list[CXEvent])
async def recent_events(last_event_id: str = "0-0") -> list[CXEvent]:
    events, _ = requires_state()
    rows = await events.read(last_event_id)
    handled = await app.state.redis.smembers(HANDLED_KEY)
    return [e.model_copy(update={"handled": e.event_id in handled}) for e in rows]


async def pop_followup(customer_id: str | None, new_score: float) -> FollowUp | None:
    """If this customer had a handled alert, compare this interaction against it."""
    if not customer_id:
        return None
    key = f"{FOLLOWUP_PREFIX}{customer_id}"
    raw = await app.state.redis.get(key)
    if not raw:
        return None
    await app.state.redis.delete(key)
    previous = json.loads(raw)
    return build_followup(previous["event_id"], previous["score"], new_score)


def to_wav_16k(src: Path) -> Path:
    """Convert browser recordings (webm/opus, mp4, ...) to 16 kHz mono WAV for both models."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed. Run `brew install ffmpeg` (or apt-get install ffmpeg).")
    dst = src.with_name(src.stem + "_16k.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        dst.unlink(missing_ok=True)
        raise RuntimeError("Could not decode the audio clip.") from exc
    return dst


@app.post("/api/analyze", response_model=CXEvent)
async def analyze_audio(
    audio: UploadFile = File(...), customer_id: str | None = Form(None)
) -> CXEvent:
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Upload an audio file.")
    body = await audio.read()
    if len(body) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {settings.max_upload_mb} MB limit.")
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(body)
        audio_path = Path(temporary.name)
    events, analyzer = requires_state()
    wav_path: Path | None = None
    try:
        wav_path = await asyncio.to_thread(to_wav_16k, audio_path)
        transcript, emotion, sentiment, fusion = await asyncio.to_thread(
            analyzer.analyze, wav_path
        )
    except Exception as exc:
        print(f"Error during audio analysis: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        audio_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    attention = build_attention(transcript, emotion, fusion)
    customer = (customer_id or "").strip()[:64] or None
    followup = await pop_followup(customer, fusion.score)
    return await events.publish(
        CXEvent(
            source="audio",
            customer_id=customer,
            transcript=transcript,
            emotion=emotion,
            sentiment=sentiment,
            fusion=fusion,
            alert=attention.summary if attention else None,
            attention=attention,
            followup=followup,
        )
    )


@app.post("/api/alerts/{event_id}/handle")
async def handle_alert(event_id: str) -> dict:
    """Mark an alert as handled and start watching that customer's next interaction."""
    events, _ = requires_state()
    event = await events.get(event_id)
    if event is None or event.attention is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    redis = app.state.redis
    await redis.sadd(HANDLED_KEY, event_id)
    if event.customer_id and event.fusion:
        await redis.set(
            f"{FOLLOWUP_PREFIX}{event.customer_id}",
            json.dumps({"event_id": event_id, "score": event.fusion.score}),
            ex=FOLLOWUP_TTL_SECONDS,
        )
    return {"event_id": event_id, "handled": True, "watching": event.customer_id}


@app.post("/api/demo/seed", response_model=DemoSeedResponse)
async def seed_demo() -> DemoSeedResponse:
    """Add clearly-labelled simulated events to verify the dashboard without customer audio."""
    events, _ = requires_state()
    samples = [
        ("I appreciate how quickly this was resolved.", "happy", 0.82, "positive", 0.99),
        ("I have been waiting all day and still need help.", "angry", 0.77, "negative", 0.98),
        ("Could you explain the next step for my account?", "calm", 0.74, "neutral", 0.68),
    ]
    published = []
    for text, emotion, emotion_score, sentiment, sentiment_score in samples:
        fused = fuse(emotion, emotion_score, sentiment, sentiment_score)
        fusion = FusionResult(
            state=fused.state,
            score=fused.score,
            sentiment_score=fused.sentiment_score,
            emotion_score=fused.emotion_score,
        )
        event = CXEvent(
            source="simulated",
            transcript=text,
            emotion=EmotionResult(label=emotion, confidence=emotion_score, status="ready"),
            sentiment=SentimentResult(label=sentiment, confidence=sentiment_score),
            fusion=fusion,
        )
        published.append(await events.publish(event))
    return DemoSeedResponse(events=published)


@app.websocket("/ws/events")
async def event_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    events, _ = requires_state()
    last_id = websocket.query_params.get("last_event_id", "$")
    try:
        while True:
            # Wait for either a durable stream event or a client/server disconnect.
            # A standalone XREAD loop cannot observe a closed idle WebSocket promptly.
            stream_read = asyncio.create_task(events.read(last_id, block_ms=5_000))
            disconnect = asyncio.create_task(websocket.receive())
            done, pending = await asyncio.wait(
                {stream_read, disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if disconnect in done:
                message = disconnect.result()
                if message["type"] == "websocket.disconnect":
                    return
            if stream_read in done:
                for event in stream_read.result():
                    await websocket.send_json(event.model_dump(mode="json"))
                    last_id = event.event_id or last_id
    except WebSocketDisconnect:
        return
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
