from __future__ import annotations

import json
from collections.abc import Mapping

from app.core.errors import ResponseFailedEvent
from app.core.types import JsonValue

type JsonPayload = Mapping[str, JsonValue] | ResponseFailedEvent


def format_sse_event(payload: JsonPayload) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    event_type = payload.get("type")
    if isinstance(event_type, str) and event_type:
        return f"event: {event_type}\ndata: {data}\n\n"
    return f"data: {data}\n\n"
