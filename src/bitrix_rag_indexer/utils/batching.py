from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
