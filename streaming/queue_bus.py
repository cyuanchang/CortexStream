from queue import Empty, Full, Queue
from typing import Generic, TypeVar


T = TypeVar("T")


class QueuePublisher(Generic[T]):
    def __init__(self, maxsize: int):
        self._queue: Queue[T] = Queue(maxsize=maxsize)

    def get_queue(self) -> Queue[T]:
        return self._queue

    def publish_keep_latest(self, item: T) -> bool:
        """Publish one item and drop oldest if queue is full."""
        try:
            self._queue.put_nowait(item)
            return False
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(item)
            return True
