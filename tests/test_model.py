import pytest
from llambda.model import Expression, Function, Variable


def add(x, y):
    return x + y


def sum_(*args, **kwargs):
    return sum(args) + sum(kwargs.values())


class TestVariable:
    @pytest.mark.parametrize(
        "name, value, expected",
        [
            ("x", 1, 1),
            ("x", "a", "a"),
            ("x", [1, 2, 3], [1, 2, 3]),
            ("y", Variable("x", 1), 1),
            ("y", Expression("x + 1", [Variable("x", 1)]), 2),
            ("y", Function(lambda x: x + 1, (Variable("x", 1),), {}), 2),
        ]
    )
    def test_eval(self, name, value, expected):
        variable = Variable(name, value)
        assert variable.eval() == expected

    @pytest.mark.parametrize(
        "name, value, expected",
        [
            ("x", 1, "x"),
            ("y", "a", "y"),
            ("z", [1, 2, 3], "z"),
        ]
    )
    def test_expr(self, name, value, expected):
        variable = Variable(name, value)
        assert variable.expr() == expected

    @pytest.mark.parametrize(
        "name, value, expected",
        [
            ("x", 1, "x = 1"),
            ("y", "a", "y = 'a'"),
            ("z", [1, 2, 3], "z = [1, 2, 3]"),
            ("x", Variable("y", 1), "x = y"),
            ("x", Expression("y + 1", [Variable("y", 1)]), "x = y + 1"),
            ("x", Function(add, (Variable("y", 1), Variable("z", 2)), {}), "x = add(y, z)"),
        ]
    )
    def test_stmt(self, name, value, expected):
        variable = Variable(name, value)
        assert variable.stmt() == expected


class TestFunction:
    @pytest.mark.parametrize(
        "func, args, kwargs, expected",
        [
            (add, (1, 2), {}, 3),
            (add, (Variable("x", 1), Variable("y", 2)), {}, 3),
            (add, (Variable("x", 1), 2), {}, 3),
            (add, (Expression("x + 1", [Variable("x", 2)]), 3), {}, 6),
            (sum_, (1, 2, 3), {}, 6),
            (sum_, (Variable("x", 1), Variable("y", 2), Variable("z", 3)), {}, 6),
            (sum_, (Variable("x", 1), 2, 3), {}, 6),
            (sum_, (Expression("x + 1", [Variable("x", 2)]),), {"x": 3, "y": Expression("x + 1", [Variable("x", 2)])}, 9),
            (sum_, (Function(add, (Variable("x", 1), Variable("y", 2)), {}), 3), {}, 6),
        ]
    )
    def test_eval(self, func, args, kwargs, expected):
        function = Function(func, args, kwargs)
        assert function.eval() == expected

    @pytest.mark.parametrize(
        "func, args, kwargs, expected",
        [
            (add, (1, 2), {}, "add(1, 2)"),
            (add, (Variable("x", 1), Variable("y", 2)), {}, "add(x, y)"),
            (add, (Variable("x", 1), 2), {}, "add(x, 2)"),
            (add, (Expression("x + 1", [Variable("x", 2)]), 3), {}, "add(x + 1, 3)"),
            (sum_, (1, 2, 3), {}, "sum_(1, 2, 3)"),
            (sum_, (Variable("x", 1), Variable("y", 2), Variable("z", 3)), {}, "sum_(x, y, z)"),
            (sum_, (Variable("x", 1), 2, 3), {}, "sum_(x, 2, 3)"),
            (sum_, (Expression("x + 1", [Variable("x", 2)]),), {"x": 3, "y": Expression("x + 1", [Variable("x", 2)])}, "sum_(x + 1, x=3, y=x + 1)"),
            (sum_, (Function(add, (Variable("x", 1), Variable("y", 2)), {}), 3), {}, "sum_(add(x, y), 3)"),
        ]
    )
    def test_expr(self, func, args, kwargs, expected):
        function = Function(func, args, kwargs)
        assert function.expr() == expected


class TestExpression:
    @pytest.mark.parametrize(
        "_expr, variables, expected",
        [
            ("x + 1", [Variable("x", 1)], 2),
            ("x + y", [Variable("x", 1), Variable("y", 2)], 3),
            ("add_(x, y)", [Variable("add_", add), Variable("x", 1), Variable("y", 2)], 3),
            ("sum_(x, y)", [Variable("sum_", sum_), Variable("x", Function(add, (1, 2), {})), Variable("y", 3)], 6),
            ("'A'.lower()", [], "a"),
            ("x.upper()", [Variable("x", "a")], "A"),
            ("a[0]", [Variable("a", [10, 20, 30])], 10),
            (
                "a[add(sum_(x, y), 1)]",
                [
                    Variable("a", [10, 20, 30, 40]),
                    Variable("add", add),
                    Variable("sum_", sum_),
                    Variable("x", Expression("z + 1", [Variable("z", 0)])),
                    Variable("y", Function(add, (1, 0), {}))
                ],
                40
            ),
        ]
    )
    def test_eval(self, _expr, variables, expected):
        expression = Expression(_expr, variables)
        assert expression.eval() == expected
