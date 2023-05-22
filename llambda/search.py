from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from inspect import getsource
from typing import Callable

import numpy as np

from llambda.llm.base import BaseEmbeddingModel


@dataclass
class Searcher:
    functions: list[Callable]
    model: BaseEmbeddingModel
    embeddings: np.ndarray | None = None
    n_results: int = 10
    max_requests_per_second: int = 10

    async def _embed_functions(self) -> np.ndarray:
        sem = asyncio.Semaphore(self.max_requests_per_second)
        arrays = await asyncio.gather(*[
            self._embed_and_normalize(func, sem, 1)
            for func in self.functions
        ])
        return np.array(arrays)

    async def _embed_and_normalize(self, func, sem: asyncio.Semaphore, min_wait_seconds):
        async with sem:
            tasks = [
                self._embed_text(func.__name__, min_wait_seconds),
                self._embed_text(func.__doc__, min_wait_seconds),
                self._embed_text(getsource(func), min_wait_seconds),
            ]
            return np.array([normalize(vec) for vec in await asyncio.gather(*tasks)])

    async def _embed_text(self, text: str, min_wait_seconds=0) -> list[float]:
        start_time = time.time()
        res = await self.model.embed(text)
        end_time = time.time()
        if end_time - start_time < min_wait_seconds:
            await asyncio.sleep(min_wait_seconds - (end_time - start_time))
        return res

    def search(self, query) -> list[Callable]:
        if self.embeddings is None:
            self.embeddings = asyncio.run(self._embed_functions())
        query_embedding = normalize(asyncio.run(self._embed_text(query)))
        return [
            self.functions[i] for i
            in similarity_argsort(query_embedding, self.embeddings)[:self.n_results]
        ]


def normalize(vec) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def similarity_argsort(
    query_embedding: np.ndarray,
    func_embeddings: np.ndarray,
) -> list[int]:
    similarities = np.dot(func_embeddings, query_embedding)
    max_similarities = np.max(similarities, axis=1)
    return np.argsort(-max_similarities).tolist()
