from __future__ import annotations

from dataclasses import dataclass
from queue import Full, Queue
import threading
from typing import Dict, Generic, Literal, TypeVar


T = TypeVar("T")
DropPolicy = Literal["keep_latest", "drop_new"]


@dataclass(frozen=True)
class SubscriberConfig:
    name: str
    maxsize: int
    drop_policy: DropPolicy = "keep_latest"


@dataclass
class _SubscriberState(Generic[T]):
    queue: Queue[T]
    config: SubscriberConfig
    dropped: int = 0


class PubSubBus(Generic[T]):
    """Fanout bus with per-subscriber bounded queues and drop policies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[str, _SubscriberState[T]] = {}

    def subscribe(
        self,
        name: str,
        maxsize: int,
        drop_policy: DropPolicy = "keep_latest",
    ) -> Queue[T]:
        config = SubscriberConfig(name=name, maxsize=maxsize, drop_policy=drop_policy)
        with self._lock:
            existing = self._subscribers.get(name)
            if existing is not None:
                return existing.queue
            queue: Queue[T] = Queue(maxsize=maxsize)
            self._subscribers[name] = _SubscriberState(queue=queue, config=config)
            return queue

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subscribers.pop(name, None)

    def publish(self, item: T) -> int:
        """Publish one item to all subscribers, returning total dropped deliveries."""
        with self._lock:
            states = list(self._subscribers.values())
        dropped_total = 0
        for state in states:
            dropped_total += self._publish_to_subscriber(state, item)
        return dropped_total

    def get_queue(self, name: str) -> Queue[T] | None:
        with self._lock:
            state = self._subscribers.get(name)
            if state is None:
                return None
            return state.queue

    def get_queue_size(self, name: str) -> int:
        with self._lock:
            state = self._subscribers.get(name)
            if state is None:
                return 0
            queue = state.queue
        return queue.qsize()

    def get_dropped_count(self, name: str) -> int:
        with self._lock:
            state = self._subscribers.get(name)
            if state is None:
                return 0
            return state.dropped

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def _publish_to_subscriber(self, state: _SubscriberState[T], item: T) -> int:
        if state.config.drop_policy == "drop_new":
            try:
                state.queue.put_nowait(item)
                return 0
            except Full:
                with self._lock:
                    live = self._subscribers.get(state.config.name)
                    if live is not None:
                        live.dropped += 1
                return 1

        try:
            state.queue.put_nowait(item)
            return 0
        except Full:
            try:
                state.queue.get_nowait()
            except Exception:
                # Queue became empty after Full due to races, continue with put retry.
                pass
            try:
                state.queue.put_nowait(item)
            except Full:
                with self._lock:
                    live = self._subscribers.get(state.config.name)
                    if live is not None:
                        live.dropped += 1
                return 1
            with self._lock:
                live = self._subscribers.get(state.config.name)
                if live is not None:
                    live.dropped += 1
            return 1
