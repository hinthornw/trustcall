# mypy: ignore-errors
from typing import Optional, Tuple

import pytest
from pydantic import ValidationError
from typing_extensions import Annotated, TypedDict

from trustcall._base import _convert_any_typed_dicts_to_pydantic, _apply_message_ops, MessageOp
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


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


def test_message_ops_update_tool_name():
    """Test various scenarios for updating tool names in messages."""
    
    # Test case 1: Mixed message types
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="user message"),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tool1",
                "name": "old_name",
                "args": {"arg1": "value1"}
            }]
        ),
        ToolMessage(
            content="tool response",
            tool_call_id="tool1",
            name="old_name"
        )
    ]

    message_ops = [
        MessageOp(
            op="update_tool_name",
            target={
                "id": "tool1",
                "name": "new_name"
            }
        )
    ]

    result = _apply_message_ops(messages, message_ops)

    # Verify message count and types
    assert len(result) == 4, "All messages should be preserved"
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    
    # Verify content preservation
    assert result[0].content == "system message"
    assert result[1].content == "user message"
    assert result[2].tool_calls[0]["name"] == "new_name"
    assert result[3].content == "tool response"

    # Test case 2: Multiple tool calls in single AIMessage
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool1",
                    "name": "old_name1",
                    "args": {"arg1": "value1"}
                },
                {
                    "id": "tool2",
                    "name": "old_name2",
                    "args": {"arg2": "value2"}
                }
            ]
        )
    ]

    message_ops = [
        MessageOp(
            op="update_tool_name",
            target={
                "id": "tool1",
                "name": "new_name1"
            }
        )
    ]

    result = _apply_message_ops(messages, message_ops)
    
    # Verify selective update
    assert len(result) == 1
    assert result[0].tool_calls[0]["name"] == "new_name1"  # Updated
    assert result[0].tool_calls[1]["name"] == "old_name2"  # Unchanged

    # Test case 3: No matching tool_id
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tool1",
                "name": "old_name",
                "args": {"arg1": "value1"}
            }]
        )
    ]

    message_ops = [
        MessageOp(
            op="update_tool_name",
            target={
                "id": "non_existent_tool",
                "name": "new_name"
            }
        )
    ]

    result = _apply_message_ops(messages, message_ops)
    
    # Verify no changes for non-matching tool_id
    assert result[0].tool_calls[0]["name"] == "old_name"
