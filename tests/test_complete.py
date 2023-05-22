import pytest
from llambda.complete import ContextVar, commentout


class TestContextVar:
    @pytest.mark.parametrize(
        "value, type_name, expected",
        [
            (1, "int", "int"),
            ("a", None, "str"),
            ([1, 2, 3], None, "list"),
            ({"a": 1, "b": 2}, None, "dict"),
            (lambda x: x + 1, None, "function"),
        ]
    )
    def test_type(self, value, type_name, expected):
        var = ContextVar("x", value, type_name)
        assert var.type == expected

    @pytest.mark.parametrize(
        "name, value, type_name, description, expected",
        [
            ("x", 1, "int", None, "x: int = 1"),
            ("y", "a", None, None, "y: str = a"),
            ("z", [1, 2, 3], None, None, "z: list = [1, 2, 3]"),
            ("x", 1, "int", "x is an integer", "# x is an integer\nx: int = 1"),
            ("y", "a", None, "y is a string", "# y is a string\ny: str = a"),
            ("z", [1, 2, 3], None, "z is a list", "# z is a list\nz: list = [1, 2, 3]"),
        ]
    )
    def test_stmt(self, name, value, type_name, description, expected):
        var = ContextVar(name, value, type_name, description)
        assert var.stmt() == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("a", "# a"),
        ("a\nb", "# a\n# b"),
        ("a\nb\nc", "# a\n# b\n# c"),
        ("a\n\nb", "# a\n\n# b"),
        ("a\n\n#b, ", "# a\n\n# #b, "),
        ("a\n\n#b, \n", "# a\n\n# #b, \n"),
    ]
)
def test_commentout(text, expected):
    assert commentout(text) == expected
