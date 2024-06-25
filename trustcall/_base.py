"""Utilities for tool calling and extraction with retries."""

from __future__ import annotations

import logging
import operator
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import jsonpatch  # type: ignore[import-untyped]
import langsmith
from dydantic import create_model_from_schema
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import create_schema_from_function
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.constants import Send
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ValidationNode
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import (
    MessageLikeRepresentation,
)


logger = logging.getLogger("extraction")


TOOL_T = Union[BaseTool, Type[BaseModel], Callable, Dict[str, Any]]
DEFAULT_MAX_ATTEMPTS = 3

Message = Union[MessageLikeRepresentation, MessageLikeRepresentation]

Messages = Union[Message, Sequence[Message]]


class ExtractionInputs(TypedDict, total=False):
    messages: Union[Messages, PromptValue]
    existing: Optional[Dict[str, Any]]


class ExtractionOutputs(TypedDict):
    messages: List[AIMessage]
    responses: List[BaseModel]


def create_extractor(
    llm: BaseChatModel,
    *,
    tools: Sequence[TOOL_T],
    tool_choice: Optional[str] = None,
) -> Runnable[ExtractionInputs, ExtractionOutputs]:
    """Binds validators + retry logic ensure validity of generated tool calls.

    This method is similar to `bind_validator_with_retries`, but uses JSONPatch to correct
    validation errors caused by passing in incorrect or incomplete parameters in a previous
    tool call. This method requires the 'jsonpatch' library to be installed.

    Using patch-based function healing can be more efficient than repopulating the entire
    tool call from scratch, and it can be an easier task for the LLM to perform, since it typically
    only requires a few small changes to the existing tool call.

    Args:
        llm (Runnable): The llm that will generate the initial messages (and optionally fallba)
        tools (list): The tools to bind to the LLM.
        tool_choice (Optional[str]): The tool choice to use. Defaults to None (let the LLM choose).

    Returns:
        Runnable: A runnable that can be invoked with a list of messages and returns a single AI message.
    """
    builder = StateGraph(ExtractionState)

    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]):
        if hasattr(schema, "model_json_schema"):
            schema_ = schema.model_json_schema()
        else:
            schema_ = schema.schema()
        return (
            f"Error:\n\n```\n{repr(error)}\n```\n"
            "Expected Parameter Schema:\n\n" + f"```json\n{schema_}\n```\n"
            f"Please respond with a JSONPatch to correct the error for schema_id=[{call['id']}]."
        )

    validator = ValidationNode(
        ensure_tools(tools) + [PatchFunctionParameters],
        format_error=format_exception,
    )

    builder.add_node(
        _Extract(
            llm,
            [
                schema
                for name, schema in validator.schemas_by_name.items()
                if name != PatchFunctionParameters.__name__
            ],
            tool_choice,
        ).as_runnable()
    )
    builder.add_node(_ExtractUpdates(llm).as_runnable())
    builder.add_node(_Patch(llm).as_runnable())
    builder.add_node("validate", validator)

    def enter(state: ExtractionState) -> Literal["extract", "extract_updates"]:
        if state.get("existing"):
            return "extract_updates"
        return "extract"

    builder.add_conditional_edges(START, enter)
    builder.add_edge("extract", "validate")
    builder.add_edge("extract_updates", "validate")

    def handle_retries(
        state: ExtractionState, config: RunnableConfig
    ) -> Union[Literal["__end__"], list]:
        """After validation, decide whether to retry or end the process."""
        max_attempts = config["configurable"].get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        if state["attempts"] >= max_attempts:
            return "__end__"
        # Only continue if we need to patch the tool call
        to_send = []
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                break
            if isinstance(m, ToolMessage):
                if m.additional_kwargs.get("is_error"):
                    # Each fallback will fix at most 1 schema per time.
                    messages_for_fixing = _get_history_for_tool_call(
                        state["messages"], m.tool_call_id
                    )
                    to_send.append(
                        Send("patch", {**state, "messages": messages_for_fixing})
                    )
        return to_send

    builder.add_conditional_edges("validate", RunnableLambda(handle_retries))
    builder.add_edge("patch", "validate")
    compiled = builder.compile()

    def filter_state(state: ExtractionState) -> ExtractionOutputs:
        """Filter the state to only include the validated AIMessage + responses."""
        msg_id = state.get("msg_id")
        msg: Optional[AIMessage] = next(
            (
                m
                for m in state["messages"]
                if m.id == msg_id and isinstance(m, AIMessage)
            ),
            None,
        )
        if not msg:
            return ExtractionOutputs(messages=[], responses=[])
        responses = []
        for tc in msg.tool_calls:
            sch = validator.schemas_by_name[tc["name"]]
            responses.append(
                sch.model_validate(tc["args"])
                if hasattr(sch, "model_validate")
                else sch.parse_obj(tc["args"])
            )

        return {
            "messages": [msg],
            "responses": responses,
        }

    return compiled | filter_state


def ensure_tools(
    tools: Sequence[TOOL_T],
) -> List[Union[BaseTool, Type[BaseModel], Callable]]:
    results: list = []
    for t in tools:
        if isinstance(t, dict):
            results.append(create_model_from_schema(t))
        elif isinstance(t, (BaseTool, type)):
            results.append(t)
        elif callable(t):
            results.append(create_schema_from_function(t.__name__, t))
        else:
            raise ValueError(f"Invalid tool type: {type(t)}")
    return list(results)


## Helper functions + reducers


class _Extract:
    def __init__(
        self, llm: BaseChatModel, tools: list, tool_choice: Optional[str] = None
    ):
        self.bound_llm = llm.bind_tools(tools, tool_choice=tool_choice)

    @langsmith.traceable
    def _tear_down(self, msg: AIMessage) -> dict:
        if not msg.id:
            msg.id = str(uuid.uuid4())
        return {
            "messages": [msg],
            "attempts": 1,
            "msg_id": msg.id,
        }

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        msg = await self.bound_llm.ainvoke(state["messages"], config)
        return self._tear_down(cast(AIMessage, msg))

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        msg = self.bound_llm.invoke(state["messages"], config)
        return self._tear_down(msg)

    def as_runnable(self):
        return RunnableLambda(self.invoke, self.ainvoke, name="extract")


class _ExtractUpdates:
    """Prompt an LLM to patch an existing schema.

    We have found this to be prefereable to re-generating
    the entire tool call from scratch in several ways:

    1. Fewer output tokens.
    2. Less likely to introduce new errors or drop important information.
    3. Easier for the LLM to generate."""

    def __init__(self, llm: BaseChatModel):
        self.bound = llm.bind_tools(
            [PatchFunctionParameters], tool_choice=PatchFunctionParameters.__name__
        )

    @langsmith.traceable
    def _setup(self, state: ExtractionState):
        messages = state["messages"]
        existing = state["existing"]
        if not existing:
            raise ValueError("No existing schemas provided.")
        existing_schemas = "\n".join(
            [f"<schema id={k}>\n{v}\n</schema>" for k, v in existing.items()]
        )

        existing_msg = f"""Generate a JSONPatch to update the existing schemas.

{existing_schemas}
"""
        if isinstance(messages[0], SystemMessage):
            system_message = messages.pop(0)
            if isinstance(system_message.content, str):
                system_message.content += "\n\n" + existing_msg
            else:
                system_message.content = cast(list, system_message.content) + [
                    "\n\n" + existing_msg
                ]
        else:
            system_message = SystemMessage(content=existing_msg)
        return [system_message] + messages, existing

    @langsmith.traceable
    def _teardown(self, msg: AIMessage, existing: Dict[str, Any]):
        resolved_tool_calls = []
        for tc in msg.tool_calls:
            schema_id = tc["args"]["schema_id"]
            if target := existing.get(schema_id):
                resolved_tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=schema_id,
                        args=jsonpatch.apply_patch(target, tc["args"]["patches"]),
                    )
                )
        ai_message = AIMessage(
            content=msg.content,
            tool_calls=resolved_tool_calls,
        )
        if not ai_message.id:
            ai_message.id = str(uuid.uuid4())

        return {
            "messages": [ai_message],
            "attempts": 1,
            "msg_id": ai_message.id,
        }

    async def ainvoke(
        self, state: ExtractionState, config: RunnableConfig
    ) -> ExtractionState:
        """Generate a JSONPatch to simply update an existing schema.

        Returns a single AIMessage with the updated schema, as if
            the schema were extracted from scratch.
        """
        messages, existing = self._setup(state)
        msg = await self.bound.ainvoke(messages, config)
        return self._teardown(cast(AIMessage, msg), existing)

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> ExtractionState:
        messages, existing = self._setup(state)
        msg = self.bound.invoke(messages, config)
        return self._teardown(msg, existing)

    def as_runnable(self):
        return RunnableLambda(self.invoke, self.ainvoke, name="extract_updates")


class _Patch:
    """Prompt an LLM to patch an invalid schema after it receives a ValidationError.

    We have found this to be more reliable and more token-efficient than
    re-creating the entire tool call from scratch."""

    def __init__(self, llm: BaseChatModel):
        self.bound = llm.bind_tools(
            [PatchFunctionParameters], tool_choice=PatchFunctionParameters.__name__
        )

    @langsmith.traceable
    def _tear_down(self, msg: AIMessage, messages: List[AnyMessage]) -> dict:
        if not msg.id:
            msg.id = str(uuid.uuid4())
        # We will directly update the messages in the state before validation.
        msg_ops = _infer_patch_message_ops(messages, msg.tool_calls)
        return {
            "messages": msg_ops,
            "attempts": 1,
        }

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Generate a JSONPatch to correct the validation error and heal the tool call.

        Assumptions:
            - We only support a single tool call to be patched.
            - State's message list's last AIMessage contains the actual schema to fix.
            - The last ToolMessage contains the tool call to fix.

        """
        msg = await self.bound.ainvoke(state["messages"], config)
        return self._tear_down(msg, state["messages"])

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        msg = self.bound.invoke(state["messages"], config)
        return self._tear_down(cast(AIMessage, msg), state["messages"])

    def as_runnable(self):
        return RunnableLambda(self.invoke, self.ainvoke, name="patch")


class JsonPatch(BaseModel):
    """A JSON Patch document represents an operation to be performed on a JSON document.

    Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
    Examples:

    ```json
    {"op": "add", "path": "/a/b/c", "patch_value": 1}
    {"op": "replace", "path": "/a/b/c", "patch_value": 2}
    {"op": "remove", "path": "/a/b/c"}
    ```
    """

    op: Literal["add", "remove", "replace"] = Field(
        ...,
        description="The operation to be performed. Must be one of 'add', 'remove', 'replace'.",
    )
    path: str = Field(
        ...,
        description="A JSON Pointer path that references a location within the target document where the operation is performed.",
    )
    value: Union[str, int, bool, float] = Field(
        ...,
        description="The value to be used within the operation. REQUIRED for 'add', 'replace', and 'test' operations.",
    )


class PatchFunctionParameters(BaseModel):
    """Respond with all JSONPatch operation to correct validation errors caused by passing in incorrect or incomplete parameters in a previous tool call."""

    schema_id: str = Field(
        ...,
        description="The ID of the function you are patching.",
    )
    reasoning: str = Field(
        ...,
        description="Think step-by-step, listing each validation error and the"
        " JSONPatch operation needed to correct it. "
        "Cite the fields in the JSONSchema you referenced in developing this plan.",
    )
    patches: list[JsonPatch] = Field(
        ...,
        description="A list of JSONPatch operations to be applied to the previous tool call's response.",
    )


class MessageOp(TypedDict):
    op: Literal["delete", "update_tool_call"]
    target: Union[str, ToolCall]


def _get_history_for_tool_call(messages: List[AnyMessage], tool_call_id: str):
    results = []
    seen_ai_message = False
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if not seen_ai_message:
                tool_calls = [tc for tc in m.tool_calls if tc["id"] == tool_call_id]
                m = AIMessage(
                    **m.dict(exclude={"tool_calls"}),
                    tool_calls=tool_calls,
                )
            seen_ai_message = True
        if isinstance(m, ToolMessage):
            if m.tool_call_id != tool_call_id and not seen_ai_message:
                continue
        results.append(m)
    return list(reversed(results))


def _reduce_messages(
    left: Optional[List[AnyMessage]],
    right: Union[
        AnyMessage,
        List[Union[AnyMessage, MessageOp]],
        List[BaseMessage],
        PromptValue,
        MessageOp,
    ],
) -> Messages:
    if not left:
        left = []
    if isinstance(right, PromptValue):
        right = right.to_messages()
    message_ops = []
    if isinstance(right, dict) and right.get("op"):
        message_ops = [right]
        right = []
    if isinstance(right, list):
        right_ = []
        for r in right:
            if isinstance(r, dict) and r.get("op"):
                message_ops.append(r)
            else:
                right_.append(r)
        right = right_  # type: ignore[assignment]
    messages = add_messages(left, right)  # type: ignore[arg-type]
    if message_ops:
        # Apply operations to the messages
        for message_op in message_ops:
            if message_op["op"] == "delete":
                t = cast(str, message_op["target"])
                messages = [m for m in messages if cast(str, getattr(m, "id")) != t]
            elif message_op["op"] == "update_tool_call":
                targ = cast(ToolCall, message_op["target"])
                for m in messages:
                    if isinstance(m, AIMessage):
                        old = m.tool_calls.copy()
                        m.tool_calls = [
                            targ if tc["id"] == targ["id"] else tc
                            for tc in m.tool_calls
                        ]
                        if not old == m.tool_calls:
                            break

            else:
                raise ValueError(f"Invalid operation: {message_op['op']}")
    return messages


@langsmith.traceable
def _get_message_op(messages: Sequence[AnyMessage], tool_call: dict) -> List[MessageOp]:
    target_id = tool_call["schema_id"]
    msg_ops: List[MessageOp] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] == target_id:
                    patched_args = jsonpatch.apply_patch(
                        tc["args"], tool_call["patches"]
                    )
                    msg_ops.append(
                        {
                            "op": "update_tool_call",
                            "target": {
                                "id": target_id,
                                "name": tc["name"],
                                "args": patched_args,
                            },
                        }
                    )
        if isinstance(m, ToolMessage):
            if m.tool_call_id == target_id:
                msg_ops.append(MessageOp(op="delete", target=m.id or ""))
    return msg_ops


@langsmith.traceable
def _infer_patch_message_ops(
    messages: Sequence[AnyMessage], tool_calls: List[ToolCall]
):
    return [
        op
        for tool_call in tool_calls
        for op in _get_message_op(messages, tool_call["args"])
    ]


class ExtractionState(TypedDict):
    messages: Annotated[List[AnyMessage], _reduce_messages]
    attempts: Annotated[int, operator.add]
    msg_id: Annotated[str, lambda left, right: left if left else right]
    """Set once and never changed. The ID of the message to be patched."""
    existing: Optional[Dict[str, Any]]
    """If you're updating an existing schema, provide the existing schema here."""


__all__ = [
    "create_extractor",
    "ensure_tools",
    "ExtractionInputs",
    "ExtractionOutputs",
]
