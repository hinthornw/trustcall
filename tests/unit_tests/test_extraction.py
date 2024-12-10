import uuid
from typing import Any, Callable, Dict, List, Optional

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from trustcall._base import (
    PatchDoc,
    PatchFunctionErrors,
    SchemaInstance,
    _ExtractUpdates,
    create_extractor,
    ensure_tools,
)


class FakeExtractionModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    responses: List[AIMessage] = []
    backup_responses: List[AIMessage] = []
    i: int = 0
    bound_count: int = 0
    tools: list = []

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses[self.i % len(self.responses)]
        self.i += 1
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"key": "fake"}

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeExtractionModel":  # type: ignore
        """Bind tools to the model."""
        tools = [convert_to_openai_tool(t) for t in tools]
        responses = (
            self.responses
            if self.bound_count <= 0
            else self.backup_responses[self.bound_count - 1 :]
        )
        backup_responses = self.backup_responses if self.bound_count <= 0 else []
        self.bound_count += 1
        return FakeExtractionModel(
            responses=responses,
            backup_responses=backup_responses,
            tools=tools,
            i=self.i,
            **kwargs,
        )


class MyNestedSchema(BaseModel):
    """Nested schema for testing."""

    field1: str
    """Field 1."""
    some_int: int
    """Some integer."""
    some_float: float


def my_cool_tool(arg1: str, arg2: MyNestedSchema) -> None:
    """This is a cool tool."""
    pass


def _get_tool_as(style: str) -> Any:
    """Coerce a string to a function, tool, schema, or model."""
    tool_: BaseTool = tool(my_cool_tool)  # type: ignore

    def my_cool_injected_tool(
        arg1: str,
        arg2: MyNestedSchema,
        other_arg: Annotated[str, InjectedToolArg] = "default",
    ) -> None:
        """This is a cool tool."""
        pass

    class my_cool_tool2(TypedDict):
        """This is a cool tool."""

        arg1: str
        arg2: MyNestedSchema

    setattr(my_cool_injected_tool, "__name__", "my_cool_tool")
    setattr(my_cool_tool2, "__name__", "my_cool_tool")
    if style == "fn":
        return my_cool_tool
    elif style == "tool":
        return tool_
    elif style == "schema":
        return tool_.args_schema.schema()  # type: ignore
    elif style == "model":
        return tool_.args_schema
    elif style == "typeddict":
        return my_cool_tool2
    elif style == "injected_fn":
        return my_cool_injected_tool
    elif style == "injected_tool":
        return tool(my_cool_injected_tool)
    else:
        raise ValueError(f"Invalid style: {style}")


def _get_tool_name(style: str) -> str:
    """Get the name of the tool."""
    tool_ = ensure_tools([_get_tool_as(style)])[0]
    try:
        return FakeExtractionModel().bind_tools([tool_]).tools[0]["function"]["name"]
    except Exception:
        return getattr(tool_, "__name__", "my_cool_tool")


@pytest.fixture
def expected() -> dict:
    return {
        "arg1": "This is a string.",
        "arg2": {
            "field1": "This is another string.",
            "some_int": 42,
            "some_float": 3.14,
        },
    }


@pytest.fixture
def initial() -> dict:
    return {
        "arg1": "This is a string.",
        "arg2": {
            "field1": "This is another string.",
            "some_int": "not fourty two",
            "some_float": 3.14,
        },
    }


def good_patch(tc_id: str) -> dict:
    return {
        "json_doc_id": tc_id,
        "reasoning": "because i said so.",
        "patches": [
            {"op": "replace", "path": "/arg2/some_int", "value": 42},
            {"op": "replace", "path": "/arg2/some_float", "value": 3.14},
        ],
    }


def bad_patch(tc_id: str) -> dict:
    return {
        "json_doc_id": tc_id,
        "reasoning": "because i said so.",
        "patches": [
            {"op": "replace", "path": "/arg2/some_int", "value": 42},
            {"op": "replace", "path": "/arg2/some_float", "value": "not a float"},
        ],
    }


def patch_2(tc_id: str) -> dict:
    return {
        "json_doc_id": tc_id,
        "reasoning": "because i said so.",
        "patches": [
            {"op": "replace", "path": "/arg2/some_float", "value": 3.14},
        ],
    }


@pytest.mark.parametrize(
    "style",
    [
        "typeddict",
        "fn",
        "tool",
        "schema",
        "model",
        "injected_fn",
        "injected_tool",
    ],
)
@pytest.mark.parametrize(
    "patches",
    [
        [],
        [good_patch],
        [bad_patch, patch_2],
    ],
)
@pytest.mark.parametrize("input_format", ["list", "prompt_value", "state"])
async def test_extraction_with_retries(
    style: str,
    expected: dict,
    patches: List[Callable[[str], dict]],
    initial: dict,
    input_format: str,
) -> None:
    tc_id = f"tool_{uuid.uuid4()}"
    tool_name = _get_tool_name(style)
    initial_msg = AIMessage(
        content="This is really cool ha.",
        tool_calls=[
            {"id": tc_id, "name": tool_name, "args": initial if patches else expected}
        ],
    )
    patch_messages = []
    for patch in patches:
        patch_msg = AIMessage(
            content="This is even more cool.",
            tool_calls=[
                {
                    "id": f"tool_{uuid.uuid4()}",
                    "name": PatchFunctionErrors.__name__,
                    "args": patch(tc_id),
                }
            ],
        )
        patch_messages.append(patch_msg)
    model = FakeExtractionModel(
        responses=[initial_msg], backup_responses=patch_messages, bound_count=-1
    )
    graph = create_extractor(model, tools=[_get_tool_as(style)])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a botly bot."),
            ("user", "I am a user with needs."),
        ]
    )
    if input_format == "list":
        inputs: Any = prompt.invoke({}).to_messages()
    elif input_format == "prompt_value":
        inputs = prompt.invoke({})
    else:
        inputs = {"messages": prompt.invoke({}).to_messages()}

    res = await graph.ainvoke(inputs)
    assert len(res["messages"]) == 1

    msg = res["messages"][0]
    assert msg.content == initial_msg.content
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["id"] == tc_id
    assert msg.tool_calls[0]["name"] == tool_name
    assert msg.tool_calls[0]["args"] == expected
    tool_: BaseTool = tool(my_cool_tool)  # type: ignore
    assert len(res["responses"]) == 1
    pred = res["responses"][0].dict()
    expected_res = tool_.args_schema.validate(expected).dict()  # type: ignore
    if "injected" in style:
        expected_res["other_arg"] = "default"
        pred["other_arg"] = "default"
    assert pred == expected_res


def empty_patch(tc_id: str) -> dict:
    return {
        "json_doc_id": tc_id,
        "reasoning": "because i said so.",
        "patches": [],
    }


@pytest.mark.parametrize(
    "style",
    [
        "fn",
        "tool",
        "schema",
        "model",
    ],
)
@pytest.mark.parametrize(
    "is_valid,existing",
    [
        (
            True,
            {
                "arg1": "This is a string.",
                "arg2": {
                    "field1": "This is another string.",
                    "some_int": 42,
                    "some_float": 3.14,
                },
            },
        ),
        (
            False,
            {
                "arg1": "This is a string.",
                "arg2": {
                    "field1": "This is another string.",
                    "some_int": 42,
                    # Test that it's OK even if the initial value is incorrect.
                    "some_float": "This isn't actually correct!",
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "patches",
    [
        [empty_patch],
        [good_patch],
        [bad_patch, patch_2],
        [bad_patch, empty_patch, patch_2],
    ],
)
async def test_patch_existing(
    style: str,
    is_valid: bool,
    existing: dict,
    patches: List[Callable[[str], dict]],
    expected: dict,
) -> None:
    if not is_valid and len(patches) == 1 and patches[0] == empty_patch:
        pytest.skip("No patches to test with invalid initial.")
    tool_name = _get_tool_name(style)
    patch_messages = []
    tc_id = f"tool_{uuid.uuid4()}"
    for i, patch in enumerate(patches):
        json_doc_id = tool_name if i == 0 else tc_id
        patch_msg = AIMessage(
            content="This is even more cool.",
            tool_calls=[
                {
                    "id": tc_id if i == 0 else f"tool_{uuid.uuid4()}",
                    "name": PatchDoc.__name__,
                    "args": patch(json_doc_id),
                }
            ],
        )
        patch_messages.append(patch_msg)

    model = FakeExtractionModel(
        backup_responses=patch_messages,
    )
    graph = create_extractor(model, tools=[_get_tool_as(style)])
    res = await graph.ainvoke(
        {
            "messages": [
                ("system", "You are a botly bot."),
            ],
            "existing": {tool_name: existing},
        }
    )

    assert len(res["messages"]) == 1
    msg = res["messages"][0]
    assert msg.content == "This is even more cool."
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["id"] == tc_id
    assert msg.tool_calls[0]["name"] == tool_name
    assert msg.tool_calls[0]["args"] == expected
    tool_: BaseTool = tool(my_cool_tool)  # type: ignore
    assert len(res["responses"]) == 1
    assert res["responses"][0].dict() == tool_.args_schema.validate(expected).dict()  # type: ignore


@pytest.mark.parametrize(
    "existing, tools, is_valid",
    [
        ({"tool1": {"key": "value"}}, {"tool1": BaseModel}, True),
        ({"invalid_tool": {"key": "value"}}, {"tool1": BaseModel}, False),
        (
            [SchemaInstance("id1", "tool1", {"key": "value"})],
            {"tool1": BaseModel},
            True,
        ),
        (
            [SchemaInstance("id1", "invalid_tool", {"key": "value"})],
            {"tool1": BaseModel},
            False,
        ),
        ([("id1", "tool1", {"key": "value"})], {"tool1": BaseModel}, True),
        ([("id1", "invalid_tool", {"key": "value"})], {"tool1": BaseModel}, False),
        ("invalid_type", {"tool1": BaseModel}, False),
    ],
)
def test_validate_existing(existing, tools, is_valid):
    extractor = _ExtractUpdates(FakeExtractionModel(), tools=tools)

    if is_valid:
        extractor._validate_existing(existing)
    else:
        with pytest.raises(ValueError):
            extractor._validate_existing(existing)
