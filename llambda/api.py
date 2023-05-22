from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable

from llambda.llm import OpenAIChatCompletion, OpenAIEmbedding
from llambda.llm.base import BaseCompletionModel, BaseEmbeddingModel
from llambda.model import Function
from llambda.search import Searcher
from llambda.select import ContextVar, Selector

registered_funcs: defaultdict[str, list[Callable]] = defaultdict(list)


def register(label_or_func: str | Callable = "default") -> Callable:
    """
    Decorator to register a function with a label.

    If this is used without parentheses, the function will be registered with the "default" label.
    """
    if callable(label_or_func):
        return register()(label_or_func)
    label = label_or_func
    def _register(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        registered_funcs[label].append(wrapper)
        return wrapper
    return _register


class ContextVars:
    """
    Base class for defining context variables.

    Each class variable defined in a subclass will be automatically recognized as a context variable.
    """
    def get_variables(self) -> list[ContextVar]:
        """
        Collect the context variables defined as properties or class variables in the class.
        """
        variables: list[ContextVar]= []
        var_names = (
            name for name in vars(self.__class__).keys()
            if not name.startswith("_") and name != "get_variables"
        )
        for name in var_names:
            if isinstance(getattr(self.__class__, name), property):
                prop = getattr(self.__class__, name)
                value = getattr(self, name)
                type_annotation = prop.fget.__annotations__.get("return")
                if type_annotation and not isinstance(type_annotation, str):
                    type_annotation = type_annotation.__name__
                comment = prop.fget.__doc__
                variables.append(ContextVar(name, value, type_annotation, comment))
            else:
                value = getattr(self, name)
                type_annotation = self.__class__.__annotations__.get(name)
                if type_annotation and not isinstance(type_annotation, str):
                    type_annotation = type_annotation.__name__
                variables.append(ContextVar(name, value, type_annotation, None))
        return variables


@dataclass
class LLambda:
    """
    Main class of LLambda.

    An instance of this class takes natural language instructions and executes the corresponding function.
    """
    searcher: Searcher
    selector: Selector
    variables: list[ContextVar] = field(default_factory=list)

    def __call__(self, instruction: str):
        return self.select(instruction)()

    def search(self, instruction: str) -> list[Callable]:
        """
        Search for functions corresponding to the instruction.
        """
        if self.searcher.n_results >= len(self.searcher.functions):
            return self.searcher.functions
        return self.searcher.search(instruction)

    def select(self, instruction: str) -> Function:
        """
        Select the best function from the searched functions and generate properly arguments
        corresponding to the instruction.
        """
        searched_funcs = self.search(instruction)
        aliased_funcs = {}
        for func in searched_funcs:
            alias = func.__name__
            while alias in aliased_funcs:
                alias += "_"
            aliased_funcs[alias] = func
        return self.selector.select(instruction, aliased_funcs, self.variables)


def create_llambda(
        func: Callable | list[Callable] | None = None,
        context: list[ContextVar] | ContextVars | type[ContextVars] | None = None,
        embedding_model: BaseEmbeddingModel = OpenAIEmbedding(),
        completion_model: BaseCompletionModel = OpenAIChatCompletion(),
        register_label: str | list[str] = "default",
        n_search_results: int = 10,
) -> LLambda:
    """
    Create an instance of LLambda.

    Args:
        func: The functions to be registered.
              It can be a single function, a list of functions, or None if you use the register decorator.
        context: The context variables.
                 It can be a list of ContextVar instances, a ContextVars subclass,
                 an instance of a ContextVars subclass, or None if you don't need any context variables.
        embedding_model: An instance of BaseEmbeddingModel for the function search.
                         By default, OpenAIEmbedding is used.
        completion_model: An instance of BaseCompletionModel for function selection.
                          By default, OpenAIChatCompletion is used.
        register_label: The label(s) to use to register functions using the register decorator.
                        By default, "default" is used.
        n_search_results: The number of search results to return. By default, 10 results are returned.

    Returns:
        An instance of LLambda.
    """
    if func is None:
        funcs = []
    elif isinstance(func, list):
        funcs = func
    elif callable(func):
        funcs = [func]
    else:
        raise TypeError(f"func must be a list of functions or a single function, not {type(func)}")

    for label in register_label if isinstance(register_label, list) else [register_label]:
        funcs = registered_funcs[label] + funcs

    if context is None:
        variables = []
    elif isinstance(context, list):
        variables = context
    elif isinstance(context, ContextVars):
        variables = context.get_variables()
    elif isinstance(context, type) and issubclass(context, ContextVars):
        variables = context().get_variables()
    else:
        raise TypeError(f"context must be a list of ContextVar or a ContextVars class or instance, not {type(context)}")

    searcher = Searcher(funcs, embedding_model, n_results=n_search_results)
    selector = Selector(completion_model)
    return LLambda(searcher, selector, variables)
