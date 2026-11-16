"""Run independent model calls side by side and keep the results in input order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int = 1,
    on_done: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Apply ``fn`` to every item and return the results in the order of ``items``.

    ``workers=1`` runs one call at a time, exactly like a for loop. More workers
    only change how long the run takes: each question is handled on its own, so
    the results do not depend on the number of workers. ``on_done(done, total)``
    is called after each finished item.
    """

    items = list(items)
    total = len(items)
    if workers <= 1 or total <= 1:
        results = []
        for item in items:
            results.append(fn(item))
            if on_done:
                on_done(len(results), total)
        return results

    slots: list = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                slots[futures[future]] = future.result()
                if on_done:
                    on_done(done, total)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return slots


class TokenBudget:
    """Keeps a run under a provider's tokens-per-minute limit.

    Every call waits until the budget is positive, then pays for the tokens it
    actually used. The budget refills at the per-minute rate, so many workers
    can share one limit without triggering rate-limit errors.
    """

    def __init__(self, tokens_per_minute: int, burst_seconds: float = 6.0) -> None:
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")
        self.rate = tokens_per_minute / 60.0
        self.capacity = self.rate * burst_seconds
        self.available = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.available = min(self.capacity, self.available + (now - self._updated) * self.rate)
        self._updated = now

    def wait(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self.available > 0:
                    return
                delay = -self.available / self.rate
            time.sleep(min(delay, 5.0) + 0.01)

    def spend(self, tokens: int) -> None:
        with self._lock:
            self._refill()
            self.available -= tokens
