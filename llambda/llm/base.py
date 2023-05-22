from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCompletionModel(ABC):
    @abstractmethod
    def complete(self, query: str) -> str:
        pass


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass
