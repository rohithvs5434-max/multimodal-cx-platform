import json
from collections.abc import Sequence

from redis.asyncio import Redis

from app.schemas import CXEvent


class EventStream:
    """Redis Streams retain events, so a reconnecting dashboard can catch up."""

    def __init__(self, client: Redis, key: str, retention: int) -> None:
        self.client = client
        self.key = key
        self.retention = retention

    async def publish(self, event: CXEvent) -> CXEvent:
        event_id = await self.client.xadd(
            self.key, {"event": event.model_dump_json()}, maxlen=self.retention, approximate=True
        )
        return event.model_copy(update={"event_id": event_id})

    @staticmethod
    def _decode(rows: Sequence[tuple[str, dict[str, str]]]) -> list[CXEvent]:
        events: list[CXEvent] = []
        for event_id, fields in rows:
            event = CXEvent.model_validate_json(fields["event"])
            events.append(event.model_copy(update={"event_id": event_id}))
        return events

    async def read(self, after: str = "0-0", block_ms: int | None = None) -> list[CXEvent]:
        rows = await self.client.xread({self.key: after}, count=100, block=block_ms)
        if not rows:
            return []
        _, stream_rows = rows[0]
        return self._decode(stream_rows)

    async def get(self, event_id: str) -> CXEvent | None:
        rows = await self.client.xrange(self.key, min=event_id, max=event_id, count=1)
        events = self._decode(rows)
        return events[0] if events else None
