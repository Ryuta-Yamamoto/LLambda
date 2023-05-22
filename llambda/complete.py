from __future__ import annotations

import re
from dataclasses import dataclass
from inspect import getsource
from typing import Any, Callable

from llambda.llm.base import BaseCompletionModel


@dataclass
class ContextVar:
    name: str
    value: Any
    type_name: str | None = None
    description: str | None = None

    @property
    def type(self) -> str:
        return self.type_name or self.value.__class__.__name__

    def stmt(self) -> str:
        left = f"{self.name}: {self.type}"
        desc = commentout(self.description).strip("\n") + "\n" if self.description else ""
        return f"{desc}{left} = {self.value}"


def commentout(text: str) -> str:
    return re.sub(r"^(?!\s*$)", "# ", text, flags=re.MULTILINE)


PROMPT_TEMPLATE = """
{func_def}

{var_def}

# Complete the continuation of the code by following the instructions below, using the functions and variables provided above.
# In the continuation of the code, make only one function call and assign the return value to the 'result' variable.
# If you cannot follow the instructions, assign `NotImplementedError('Reason')` to the 'result' variable.

{instruction_comment}
result = """    # noqa


def prompt(instruction: str, funcs: list[Callable], variables: list[ContextVar]) -> str:
    func_def = "\n".join([getsource(func) for func in funcs])
    var_def = "\n".join([var.stmt() for var in variables])
    return PROMPT_TEMPLATE.format(
        func_def=func_def,
        var_def=var_def,
        instruction_comment=commentout(instruction)
    )

def complete(
        instruction: str,
        funcs: list[Callable],
        variables: list[ContextVar],
        model: BaseCompletionModel,
    ) -> str:
    prompt_ = prompt(instruction, funcs, variables)
    return model.complete(prompt_)
