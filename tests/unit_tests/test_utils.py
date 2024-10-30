from typing import Optional, Tuple

import pytest
from pydantic import ValidationError
from typing_extensions import Annotated, TypedDict

from trustcall._base import _convert_any_typed_dicts_to_pydantic


def test_convert_any_typed_dicts_to_pydantic():
    class MyType(TypedDict):
        arg1: int
        arg2: list[str]

    model = _convert_any_typed_dicts_to_pydantic(MyType)
    model(arg1=3, arg2=["foo", "bar"])
    with pytest.raises(ValidationError):
        model(arg1=3.2, arg2=["foo", "bar"])

    with pytest.raises(ValidationError):
        model(arg1=3, arg2=["foo", 2])

    class MyType2(TypedDict):
        arg: Annotated[Optional[Tuple[str, int]], "foo"]

    model = _convert_any_typed_dicts_to_pydantic(MyType2)
    model(arg=("foo", 3))
    model(arg=None)
    with pytest.raises(ValidationError):
        model(arg=("foo", 3, 3.2))

    with pytest.raises(ValidationError):
        model(arg=(3, "foo"))


def test_nested_typed_dict_conversion():
    class InnerType(TypedDict):
        inner_arg: str

    class OuterType(TypedDict):
        outer_arg: int
        nested: InnerType

    model = _convert_any_typed_dicts_to_pydantic(OuterType)

    # Valid input
    model(outer_arg=42, nested={"inner_arg": "hello"})

    # Invalid outer_arg type
    with pytest.raises(ValidationError):
        model(outer_arg="not an int", nested={"inner_arg": "hello"})

    # Invalid nested structure
    with pytest.raises(ValidationError):
        model(outer_arg=42, nested={"inner_arg": 123})  # inner_arg should be str

    # Missing nested field
    with pytest.raises(ValidationError):
        model(outer_arg=42, nested={})


def test_recursive_typed_dict_conversion():
    class RecursiveType(TypedDict):
        value: int
        next: Optional["RecursiveType"]

    model = _convert_any_typed_dicts_to_pydantic(RecursiveType)

    # Valid input with no recursion
    model(value=1, next=None)

    # Valid input with one level of recursion
    model(value=1, next={"value": 2, "next": None})

    # Valid input with multiple levels of recursion
    model(value=1, next={"value": 2, "next": {"value": 3, "next": None}})

    # Invalid type for 'value'
    with pytest.raises(ValidationError):
        model(value="not an int", next=None)

    # Invalid structure in nested 'next'
    with pytest.raises(ValidationError):
        model(value=1, next={"value": 2, "next": {"invalid": "structure"}})

    # Test with a cyclic structure (should raise an error due to max recursion)
    cyclic = {"value": 1}
    cyclic["next"] = cyclic
    with pytest.raises(ValueError):  # or RecursionError, depending on implementation
        model(**cyclic)
