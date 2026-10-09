"""In-process pub/sub for the WebSocket stream; keeps a bounded history for late joiners."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any


class Hub:
    def __init__(self, history: int = 500) -> None:
        self._subs: dict[int | None, set[asyncio.Queue]] = defaultdict(set)
        self._hist: dict[int | None, deque] = defaultdict(lambda: deque(maxlen=history))

    def subscribe(self, run_id: int | None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        for ev in list(self._hist[run_id]):
            q.put_nowait(ev)
        self._subs[run_id].add(q)
        return q

    def unsubscribe(self, run_id: int | None, q: asyncio.Queue) -> None:
        self._subs[run_id].discard(q)

    def publish(self, run_id: int, event: dict[str, Any]) -> None:
        for key in (run_id, None):  # None = firehose (all runs)
            self._hist[key].append(event)
            for q in list(self._subs[key]):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass


hub = Hub()
