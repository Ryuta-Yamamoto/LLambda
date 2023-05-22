import ast

import pytest
from llambda.build import Expression, Function, Variable, build_function, check_args_safe


def add(x, y):
    return x + y


def sum_(*args, **kwargs):
    return sum(args) + sum(kwargs.values())


@pytest.mark.parametrize(
    "text, funcs, variables, expected",
    [
        (
            "add(1, 2)",
            {"add": add},
            {},
            Function(add, (Expression("1"), Expression("2")), {})
        ),
        (
            "add(x, y)",
            {"add": add, "sum": sum_},
            {"x": 3, "y": 4},
            Function(add, (Variable("x", 3), Variable("y", 4)), {})
        ),
        (
            "sum_(1, 2, 3, a=x, b=y)",
            {"sum_": sum_},
            {"x": 4, "y": 5},
            Function(
                sum_,
                (Expression("1"), Expression("2"), Expression("3")),
                {"a": Variable("x", 4), "b": Variable("y", 5)}
            )
        ),
        (
            "sum_(x + y, lis[0], a=dic['b'])",
            {"add": add, "sum_": sum_},
            {"x": 3, "y": 4, "lis": [1, 2, 3], "dic": {"a": 1, "b": 2}},
            Function(
                sum_,
                (
                    Expression(
                        "x + y",
                        [Variable("x", 3), Variable("y", 4)]), Expression("lis[0]", [Variable("lis", [1, 2, 3])]
                    )
                ),
                {"a": Expression("dic['b']", [Variable("dic", {"a": 1, "b": 2})])}
            )
        )
    ]
)
def test_build_function(text, funcs, variables, expected):
    call = build_function(text, funcs,variables)
    assert call.expr() == expected.expr()
    assert call.eval() == expected.eval()


@pytest.mark.parametrize(
    "text, funcs, variables",
    [
        ("add(1, 2)", {}, {}),  # Function is not defined
        ("add(x, y)", {"add": add}, {"x": 3}),  # Variable y is not defined
        ("sum_(sum_(1, 2), 3)", {"sum_": sum_}, {}),  # Function call is not allowed in the argument
        ("sum_(x.real, 10)", {"sum_": sum_}, {"x": 3 + 4j}),  # Attribute is not allowed in the argument
        ("x = 1", {}, {}),  # Assign is not allowed
        ("x + y", {}, {"x": 3, "y": 4}),  # Expression except for function call is not allowed
    ]
)
def test_build_function_error(text, funcs, variables):
    with pytest.raises(ValueError):
        build_function(text, funcs, variables)



@pytest.mark.parametrize(
    "text, disallowed_ast_types",
    [
        ("add(x, y)", (ast.Call)),
        ("(1 + add(x, y)) * 5", (ast.Call)),
        ("x.y", (ast.Attribute)),
        ("(1 + x.y) * 5", (ast.Attribute)),
    ]
)
def test_check_args_safe(text, disallowed_ast_types):
    check_args_safe(ast.parse(text), ())
    with pytest.raises(ValueError):
        check_args_safe(ast.parse(text), disallowed_ast_types)
