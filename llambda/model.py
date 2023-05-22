from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable


class Evaluable(ABC):
    @abstractmethod
    def eval(self) -> Any:
        ...

    @abstractmethod
    def expr(self) -> str:
        ...

    def __str__(self) -> str:
        return self.expr()


@dataclass
class Variable(Evaluable):
    name: str
    value: Any

    def eval(self) -> Any:
        if isinstance(self.value, Evaluable):
            return self.value.eval()
        return self.value

    def expr(self) -> str:
        return self.name

    def stmt(self) -> str:
        value = self.value.expr() if isinstance(self.value, Evaluable) else repr(self.value)
        return f"{self.name} = {value}"


@dataclass
class Function(Evaluable):
    func: Callable
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def eval(self) -> Any:
        args = tuple(arg.eval() if isinstance(arg, Evaluable) else arg for arg in self.args)
        kwargs = {
            kw: arg.eval() if isinstance(arg, Evaluable)
            else arg for kw, arg in self.kwargs.items()
        }
        return self.func(*args, **kwargs)

    def expr(self) -> str:
        args = (arg.expr() if isinstance(arg, Evaluable) else repr(arg) for arg in self.args)
        kwargs = (
            f"{kw}={arg.expr() if isinstance(arg, Evaluable) else repr(arg)}"
            for kw, arg in self.kwargs.items()
        )
        args_kwargs = ", ".join(chain(args, kwargs))
        return f"{self.func.__name__}({args_kwargs})"

    def __call__(self) -> Any:
        return self.eval()


@dataclass
class Expression(Evaluable):
    _expr: str
    variables: list[Variable] = field(default_factory=list)

    def eval(self) -> Any:
        return eval(self._expr, {var.name: var.eval() for var in self.variables})

    def expr(self) -> str:
        return self._expr
