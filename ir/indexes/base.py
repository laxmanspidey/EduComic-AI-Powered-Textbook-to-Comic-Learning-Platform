"""
The interface every retriever shares.

The pipeline composes retrievers without knowing what kind they are, and the
experiment runner swaps one for another by name. Both only need `name` and
`search`, so that is all this base class requires. Keeping it this small is
deliberate: a dense retriever and an inverted index have almost nothing in
common internally, and forcing a wider interface would mean stub methods that
mean nothing for one side or the other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    """A searchable index over passage ids."""

    #: Short identifier used as the key in fusion runs, results tables and
    #: trace output. Must be stable: it ends up in saved run files.
    name: str = 'retriever'

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return [(passage_id, score), ...] best-first, at most `top_k` long.

        Implementations must break score ties deterministically (by passage id
        is the convention here) so that repeating an experiment reproduces the
        same ranking.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return 0
