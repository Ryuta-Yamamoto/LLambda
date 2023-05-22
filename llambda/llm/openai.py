from __future__ import annotations

import openai

from llambda.llm.base import BaseCompletionModel, BaseEmbeddingModel


def set_openai_api_key(key) -> None:
    openai.api_key = key


class OpenAIChatCompletion(BaseCompletionModel):
    def __init__(self, model: str = "gpt-3.5-turbo", **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def complete(self, query: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            **self.kwargs,
        )
        return response["choices"][0]["message"]["content"] # type: ignore


class OpenAIEmbedding(BaseEmbeddingModel):
    def __init__(self, model: str = "text-embedding-ada-002", **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    async def embed(self, text: str) -> list[float]:
        response = await openai.Embedding.acreate(
            input=text,
            model=self.model,
            **self.kwargs,
        )
        return response["data"][0]["embedding"] # type: ignore
