"""Small in-process caches. Keys are always masked text, never personal data."""
from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRU(Generic[K, V]):
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._d: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: K, value: V) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)
