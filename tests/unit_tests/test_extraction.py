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
    bound: Optional["FakeExtractionModel"] = None
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
            bound=self,
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
        return tool_.args_schema.model_json_schema()  # type: ignore
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
    pred = res["responses"][0].model_dump()
    expected_res = tool_.args_schema.model_validate(expected).model_dump()  # type: ignore
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
    assert (
        res["responses"][0].model_dump()
        == tool_.args_schema.model_validate(expected).model_dump()  # type: ignore
    )  # type: ignore


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


@pytest.mark.parametrize("strict_mode", [True, False, "ignore"])
async def test_e2e_existing_schema_policy_behavior(strict_mode):
    class MyRecognizedSchema(BaseModel):
        """A recognized schema that the pipeline can handle."""

        user_id: str  # type: ignore
        notes: str  # type: ignore

    # Our existing data includes 2 top-level keys: recognized, unknown
    existing_schemas = {
        "MyRecognizedSchema": {"user_id": "abc", "notes": "original notes"},
        "UnknownSchema": {"random_field": "???"},
    }

    # The AI's single message calls PatchDoc on both recognized + unknown
    recognized_patch_id = str(uuid.uuid4())
    unknown_patch_id = str(uuid.uuid4())

    ai_msg = AIMessage(
        content="I want to patch both recognized and unknown schema.",
        tool_calls=[
            {
                "id": recognized_patch_id,
                "name": PatchDoc.__name__,
                "args": {
                    "json_doc_id": "MyRecognizedSchema",
                    "planned_edits": "update recognized doc",
                    "patches": [
                        {"op": "replace", "path": "/notes", "value": "updated notes"},
                    ],
                },
            },
            {
                "id": unknown_patch_id,
                "name": PatchDoc.__name__,
                "args": {
                    "json_doc_id": "UnknownSchema",
                    "planned_edits": "update unknown doc",
                    "patches": [
                        {
                            "op": "replace",
                            "path": "/random_field",
                            "value": "now recognized?",
                        },
                    ],
                },
            },
        ],
    )

    # LLM returns just this single message
    fake_llm = FakeExtractionModel(responses=[ai_msg], backup_responses=[ai_msg] * 10)

    # 3. Create the extractor with recognized schema, override existing_schema_policy
    extractor = create_extractor(
        llm=fake_llm,
        tools=[MyRecognizedSchema],
        enable_inserts=False,
        existing_schema_policy=strict_mode,
    )

    inputs = {
        "messages": [
            ("system", "System instructions"),
            ("user", "Update these docs, please!"),
        ],
        "existing": existing_schemas,
    }
    if strict_mode is True:
        with pytest.raises(
            ValueError, match="Key 'UnknownSchema' doesn't match any schema"
        ):
            await extractor.ainvoke(inputs)
        return

    result = await extractor.ainvoke(inputs)
    # The pipeline returns a dict with "messages", "responses", etc.
    # We should have exactly 1 final AIMessage (the one from fake_llm).
    assert len(result["messages"]) == 1
    final_msg = result["messages"][0]
    assert isinstance(final_msg, AIMessage)

    recognized_call = next(
        (tc for tc in final_msg.tool_calls if tc["id"] == recognized_patch_id), None
    )
    assert recognized_call, "Missing recognized schema patch from final messages"
    assert recognized_call["args"]["notes"] == "updated notes", (
        "Recognized schema wasn't updated"
    )

    # For the unknown schema:
    unknown_call = next(
        (tc for tc in final_msg.tool_calls if tc["id"] == unknown_patch_id), None
    )
    if strict_mode == "ignore":
        assert unknown_call is None, (
            "Unknown schema patch should be skipped in 'ignore' mode"
        )
        return

    assert unknown_call["args"] == {"random_field": "now recognized?"}, (
        "Unknown schema should still be updated in strict_mode=False"
    )

    recognized_responses = [
        r for r in result["responses"] if getattr(r, "user_id", None) == "abc"
    ]
    assert len(result["responses"]) == 1
    assert len(recognized_responses) == 1
    recognized_item = recognized_responses[0]
    # user_id = "abc", notes = "updated notes"
    assert recognized_item.notes == "updated notes"


@pytest.mark.parametrize("strict_mode", [True, False, "ignore"])
async def test_e2e_existing_schema_policy_tuple_behavior(strict_mode):
    class MyRecognizedSchema(BaseModel):
        """A recognized schema that the pipeline can handle."""

        user_id: str  # type: ignore
        notes: str  # type: ignore

    existing_schemas = [
        (
            "rec_id_1",
            "MyRecognizedSchema",
            {"user_id": "abc", "notes": "original notes"},
        ),
        ("rec_id_2", "UnknownSchema", {"random_field": "???"}),
    ]

    recognized_patch_id = str(uuid.uuid4())
    unknown_patch_id = str(uuid.uuid4())

    ai_msg = AIMessage(
        content="I want to patch recognized and unknown schemas.",
        tool_calls=[
            {
                "id": recognized_patch_id,
                "name": PatchDoc.__name__,
                "args": {
                    "json_doc_id": "rec_id_1",
                    "planned_edits": "update recognized doc",
                    "patches": [
                        {"op": "replace", "path": "/notes", "value": "updated notes"},
                    ],
                },
            },
            {
                "id": unknown_patch_id,
                "name": PatchDoc.__name__,
                "args": {
                    "json_doc_id": "rec_id_2",
                    "planned_edits": "update unknown doc",
                    "patches": [
                        {
                            "op": "replace",
                            "path": "/random_field",
                            "value": "now recognized?",
                        },
                    ],
                },
            },
        ],
    )

    # LLM returns just this single message
    fake_llm = FakeExtractionModel(responses=[ai_msg], backup_responses=[ai_msg] * 3)

    # Create the extractor with one recognized schema, override existing_schema_policy
    extractor = create_extractor(
        llm=fake_llm,
        tools=[MyRecognizedSchema],
        enable_inserts=False,
        existing_schema_policy=strict_mode,
    )

    inputs = {
        "messages": [
            ("system", "System instructions"),
            ("user", "Update these docs, please!"),
        ],
        "existing": existing_schemas,
    }

    if strict_mode is True:
        with pytest.raises(
            ValueError, match="Unknown schema 'UnknownSchema' at index 1"
        ):
            await extractor.ainvoke(inputs)
        return

    # Otherwise, we proceed
    result = await extractor.ainvoke(inputs)
    assert len(result["messages"]) == 1
    final_msg = result["messages"][0]
    assert isinstance(final_msg, AIMessage)

    recognized_call = next(
        (tc for tc in final_msg.tool_calls if tc["id"] == recognized_patch_id), None
    )
    assert recognized_call, "Missing recognized schema patch from final messages"
    assert recognized_call["args"]["notes"] == "updated notes"

    # Confirm how unknown schema was handled
    unknown_call = next(
        (tc for tc in final_msg.tool_calls if tc["id"] == unknown_patch_id), None
    )
    if strict_mode == "ignore":
        # The unknown patch should be dropped
        assert unknown_call is None, (
            "Unknown schema patch should be skipped in 'ignore' mode"
        )
        # Only recognized schema remains
        assert len(result["responses"]) == 1
        recognized_item = result["responses"][0]
        assert recognized_item.notes == "updated notes"
        return

    # If strict_mode == False, unknown schema is carried along as a raw object
    assert unknown_call is not None
    assert unknown_call["args"] == {"random_field": "now recognized?"}
    # We do still get 1 recognized response object
    recognized_responses = [
        r for r in result["responses"] if getattr(r, "user_id", None) == "abc"
    ]
    assert len(recognized_responses) == 1
    recognized_item = recognized_responses[0]
    assert recognized_item.notes == "updated notes"


@pytest.mark.parametrize("enable_inserts", [True, False])
async def test_enable_deletes_flow(enable_inserts: bool) -> None:
    class MySchema(BaseModel):
        """Schema for recognized docs."""

        data: str

    existing_docs = [
        ("Doc1", "MySchema", {"data": "contents of doc1"}),
        ("Doc2", "MySchema", {"data": "contents of doc2"}),
    ]

    remove_doc_call_id = str(uuid.uuid4())
    remove_message = AIMessage(
        content="I want to remove Doc1",
        tool_calls=[
            {
                "id": remove_doc_call_id,
                "name": "RemoveDoc",  # This is recognized only if enable_deletes=True
                "args": {"json_doc_id": "Doc1"},
            }
        ],
    )

    fake_llm = FakeExtractionModel(
        responses=[remove_message], backup_responses=[remove_message] * 3
    )

    extractor = create_extractor(
        llm=fake_llm,
        tools=[MySchema],
        enable_inserts=enable_inserts,
        enable_deletes=True,
    )

    # Invoke the pipeline with some dummy "system" prompt and existing docs
    result = await extractor.ainvoke(
        {
            "messages": [("system", "System instructions: handle doc removal.")],
            "existing": existing_docs,
        }
    )

    # The pipeline always returns final "messages" in result["messages"].
    # Because "RemoveDoc" isn't a recognized schema in the final output,
    # we won't see it among result["responses"] either way.
    assert len(result["messages"]) == 1
    final_ai_msg = result["messages"][0]
    assert isinstance(final_ai_msg, AIMessage)

    assert len(final_ai_msg.tool_calls) == 1
    assert len(result["responses"]) == 1
    assert result["responses"][0].__repr_name__() == "RemoveDoc"  # type: ignore


def test_raises_on_nothing_enabled():
    def foo() -> None:
        """bar"""
        ...

    with pytest.raises(Exception):
        create_extractor(
            llm="openai:foo",
            tools=[foo],
            enable_inserts=False,
            enable_updates=False,
            enable_deletes=False,
        )


async def test_invalid_tool_call_handling():
    """Test that invalid tool calls in additional_kwargs are handled gracefully.

    This reproduces the issue where LLM returns invalid tool calls (e.g., due to token limits)
    that result in empty tool_calls array but invalid tool call info in additional_kwargs.
    Without proper handling, this would cause AttributeError: 'ExtractionState' object has no attribute 'tool_call_id'.
    """

    # Create a simple schema for testing
    class TestSchema(BaseModel):
        name: str
        value: int

    # Create an AIMessage that simulates the invalid tool call scenario from the JSON file
    # This mimics what happens when LLM hits token limits and returns malformed JSON
    invalid_tool_call_message = AIMessage(
        content="",  # Empty content like in the JSON file
        tool_calls=[],  # Empty tool_calls array - this is the key issue
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_invalid_test_123",
                    "function": {
                        "name": "TestSchema",
                        "arguments": '{"name": "test", "value": "invalid_json_here...',  # Malformed JSON
                    },
                    "type": "invalid_tool_call",  # This indicates parsing failure
                    "error": "Unterminated string starting at: line 1 column 64 (char 63)",
                }
            ],
            "finish_reason": "length",  # Indicates token limit was hit
        },
    )

    # Create a fake LLM that returns the invalid tool call message
    fake_llm = FakeExtractionModel(
        responses=[invalid_tool_call_message],
        backup_responses=[invalid_tool_call_message] * 3,
    )

    # Create extractor with the test schema
    extractor = create_extractor(
        llm=fake_llm,
        tools=[TestSchema],
        enable_inserts=True,
        enable_updates=True,
    )

    # This should not raise AttributeError: 'ExtractionState' object has no attribute 'tool_call_id'
    # Instead, it should handle the invalid tool call gracefully
    result = await extractor.ainvoke("Extract a test schema")

    # The result should be empty since the tool call was invalid and couldn't be processed
    assert len(result["responses"]) == 0
    assert result["attempts"] > 0  # Should have attempted to process
