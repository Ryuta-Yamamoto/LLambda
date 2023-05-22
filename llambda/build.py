from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable

from llambda.model import Evaluable, Expression, Function, Variable


@dataclass
class Call:
    func: str
    args: list[ast.AST]
    kwargs: dict[str, ast.AST]


def _parse_call(call_expr: ast.Call) -> Call:
    func = call_expr.func.id    # type: ignore
    args = tuple(call_expr.args)
    kwargs = {kw.arg: kw.value for kw in call_expr.keywords}
    return Call(func, args, kwargs) # type: ignore


def parse(text: str) -> Call:
    tree = ast.parse(text)
    if isinstance(tree.body[0].value, ast.Call):    # type: ignore
        return _parse_call(tree.body[0].value)  # type: ignore
    raise ValueError(f"The text is not a function call. {text}")


def check_args_safe(
        node: ast.AST,
        disallowed_ast_types: tuple[type[ast.AST], ...]
    ) -> None:
        """Check if the node is safe to evaluate."""
        for node_ in ast.walk(node):
            if isinstance(node_, disallowed_ast_types):
                raise ValueError(f"Function call is not allowed in the argument: {ast.dump(node_)}")


def get_variables(node: ast.AST) -> set[str]:
    """Get all variable names from the AST."""
    return {node.id for node in ast.walk(node) if isinstance(node, ast.Name)}


def ast2evaluable(node: ast.AST, variables: dict[str, Any]) -> Evaluable:
    if isinstance(node, ast.Name):
        if node.id in variables:
            return Variable(node.id, variables[node.id])
        raise ValueError(f"Variable {node.id} is not defined.")
    var_names = get_variables(node)
    if var_names - set(variables.keys()):
        raise ValueError(f"Variables {var_names - set(variables.keys())} are not defined.")
    return Expression(ast.unparse(node), [Variable(var, variables[var]) for var in var_names])


def _build_function(
    call: Call,
    funcs: dict[str, Callable],
    variables: dict[str, type],
) -> Function:
    if call.func not in funcs:
        raise ValueError(f"Function {call.func} is not defined.")
    func = funcs[call.func]
    args = tuple(ast2evaluable(arg, variables) for arg in call.args)
    kwargs = {k: ast2evaluable(v, variables) for k, v in call.kwargs.items()}
    return Function(func, args, kwargs)


def build_function(
    text: str,
    funcs: dict[str, Callable],
    variables: dict[str, type],
) -> Function:
    call = parse(text)
    for arg in chain(call.args, call.kwargs.values()):
        # function call and attribute access can be dangerous
        check_args_safe(arg, disallowed_ast_types=(ast.Call, ast.Attribute))
    return _build_function(call, {**funcs, "NotImplementedError": notimplemented}, variables)


def notimplemented(reason: str) -> None:
    raise NotImplementedError(reason)
