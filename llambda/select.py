from __future__ import annotations

from typing import Callable

from llambda.build import build_function
from llambda.complete import ContextVar, complete
from llambda.llm.base import BaseCompletionModel
from llambda.model import Function


class Selector:
    def __init__(self, model: BaseCompletionModel) -> None:
        self.model = model

    def select(self, instruction: str, funcs: dict[str, Callable], variables: list[ContextVar]) -> Function:
        call = complete(instruction, list(funcs.values()), variables, self.model)
        return build_function(call, funcs, {var.name: var.value for var in variables})
