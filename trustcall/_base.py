"""Utilities for tool calling and extraction with retries."""

from __future__ import annotations

import functools
import inspect
import logging
import operator
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import jsonpatch  # type: ignore[import-untyped]
import langsmith as ls
from dydantic import create_model_from_schema
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, create_schema_from_function
from langgraph.constants import Send
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt.tool_validator import ValidationNode, get_executor_for_config
from langgraph.utils import RunnableCallable
from typing_extensions import Annotated, TypedDict, get_args

logger = logging.getLogger("extraction")


TOOL_T = Union[BaseTool, Type[BaseModel], Callable, Dict[str, Any]]
DEFAULT_MAX_ATTEMPTS = 3

Message = Union[MessageLikeRepresentation, MessageLikeRepresentation]

Messages = Union[Message, Sequence[Message]]


class SchemaInstance(NamedTuple):
    record_id: str
    schema_name: str
    record: Dict[str, Any]


class ExtractionInputs(TypedDict, total=False):
    messages: Union[Messages, PromptValue]
    existing: Optional[Union[Dict[str, Any], List[SchemaInstance]]]
    """Existing schemas. Key is the schema name, value is the schema instance.
    If a list, supports duplicate schemas to update.
    """


InputsLike = Union[ExtractionInputs, List[AnyMessage], PromptValue]


class ExtractionOutputs(TypedDict):
    messages: List[AIMessage]
    responses: List[BaseModel]
    attempts: int


def create_extractor(
    llm: BaseChatModel,
    *,
    tools: Sequence[TOOL_T],
    tool_choice: Optional[str] = None,
    enable_inserts: bool = False,
) -> Runnable[InputsLike, ExtractionOutputs]:
    """Create an extractor that generates validated structured outputs using an LLM.

    This function binds validators and retry logic to ensure the validity of
    generated tool calls. It uses JSONPatch to correct validation errors caused
    by incorrect or incomplete parameters in previous tool calls.

    Args:
        llm (BaseChatModel): The language model that will generate the initial
            messages and fallbacks.
        tools (Sequence[TOOL_T]): The tools to bind to the LLM. Can be BaseTool,
                                Type[BaseModel], Callable, or Dict[str, Any].
        tool_choice (Optional[str]): The specific tool to use. If None,
            the LLM chooses whether to use (or not use) a tool based
            on the input messages. (default: None)
        enable_inserts (bool): Whether to allow the LLM to extract new schemas
            even if it receives existing schemas. (default: False)

    Returns:
        Runnable[ExtractionInputs, ExtractionOutputs]: A runnable that
        can be invoked with a list of messages and returns validated AI
        messages and responses.

    Examples:
        >>> from langchain_fireworks import (
        ...     ChatFireworks,
        ... )
        >>> from pydantic import (
        ...     BaseModel,
        ...     Field,
        ... )
        >>>
        >>> class UserInfo(BaseModel):
        ...     name: str = Field(description="User's full name")
        ...     age: int = Field(description="User's age in years")
        >>>
        >>> llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v2")
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[UserInfo],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "human",
        ...                 "My name is Alice and I'm 30 years old",
        ...             )
        ...         ]
        ...     }
        ... )
        >>> result["responses"][0]
        UserInfo(name='Alice', age=30)

        Using multiple tools
        >>> from typing import (
        ...     List,
        ... )
        >>>
        >>> class Preferences(BaseModel):
        ...     foods: List[str] = Field(description="Favorite foods")
        >>>
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[
        ...         UserInfo,
        ...         Preferences,
        ...     ],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "system",
        ...                 "Extract all the user's information and preferences"
        ...                 "from the conversation below using parallel tool calling.",
        ...             ),
        ...             (
        ...                 "human",
        ...                 "I'm Bob, 25 years old, and I love pizza and sushi",
        ...             ),
        ...         ]
        ...     }
        ... )
        >>> print(result["responses"])
        [UserInfo(name='Bob', age=25), Preferences(foods=['pizza', 'sushi'])]
        >>> print(result["messages"])  # doctest: +SKIP
        [
            AIMessage(
                content='', tool_calls=[
                    ToolCall(id='...', name='UserInfo', args={'name': 'Bob', 'age': 25}),
                    ToolCall(id='...', name='Preferences', args={'foods': ['pizza', 'sushi']}
                )]
            )
        ]

        Updating an existing schema:
        >>> existing = {
        ...     "UserInfo": {
        ...         "name": "Alice",
        ...         "age": 30,
        ...     },
        ...     "Preferences": {
        ...         "foods": [
        ...             "pizza",
        ...             "sushi",
        ...         ]
        ...     },
        ... }
        >>> extractor = create_extractor(
        ...     llm,
        ...     tools=[
        ...         UserInfo,
        ...         Preferences,
        ...     ],
        ... )
        >>> result = extractor.invoke(
        ...     {
        ...         "messages": [
        ...             (
        ...                 "system",
        ...                 "You are tasked with maintaining user info and preferences."
        ...                 " Use the tools to update the schemas.",
        ...             ),
        ...             (
        ...                 "human",
        ...                 "I'm Alice; just had my 31st birthday yesterday."
        ...                 " We had spinach, which is my FAVORITE!",
        ...             ),
        ...         ],
        ...         "existing": existing,
        ...     }
        ... )
    """  # noqa
    builder = StateGraph(ExtractionState)

    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]):
        return (
            f"Error:\n\n```\n{str(error)}\n```\n"
            "Expected Parameter Schema:\n\n" + f"```json\n{ _get_schema(schema)}\n```\n"
            f"Please use PatchFunctionErrors to fix all validation errors."
            f" for json_doc_id=[{call['id']}]."
        )

    validator = _ExtendedValidationNode(
        ensure_tools(tools) + [PatchDoc, PatchFunctionErrors],
        format_error=format_exception,
    )
    _extract_tools = [
        schema
        for name, schema in validator.schemas_by_name.items()
        if name not in {PatchDoc.__name__, PatchFunctionErrors.__name__}
    ]
    tool_names = [getattr(t, "name", t.__name__) for t in _extract_tools]
    builder.add_node(
        _Extract(
            llm,
            _extract_tools,
            tool_choice,
        ).as_runnable()
    )
    builder.add_node(
        _ExtractUpdates(
            llm, tools=validator.schemas_by_name.copy(), enable_inserts=enable_inserts
        ).as_runnable()
    )
    builder.add_node(_Patch(llm, valid_tool_names=tool_names).as_runnable())
    builder.add_node("validate", validator)

    def del_tool_call(state: DeletionState) -> dict:
        return {
            "messages": MessageOp(op="delete", target=state["deletion_target"]),
        }

    builder.add_node(del_tool_call)

    def enter(state: ExtractionState) -> Literal["extract", "extract_updates"]:
        if state.get("existing"):
            return "extract_updates"
        return "extract"

    builder.add_conditional_edges(START, enter)

    def validate_or_retry(
        state: ExtractionState,
    ) -> Literal["validate", "extract_updates"]:
        if state["messages"][-1].type == "ai":
            return "validate"
        return "extract_updates"

    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("extract_updates", validate_or_retry)

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
                        Send(
                            "patch",
                            {
                                **state,
                                "messages": messages_for_fixing,
                                "tool_call_id": m.tool_call_id,
                            },
                        )
                    )
                else:
                    # We want to delete the validation tool calls
                    # anyway to avoid mixing branches during fan-in
                    to_send.append(
                        Send(
                            "del_tool_call",
                            {
                                "deletion_target": m.id,
                                "messages": state["messages"],
                            },
                        )
                    )
        return to_send

    builder.add_conditional_edges(
        "validate", handle_retries, path_map=["__end__", "patch", "del_tool_call"]
    )

    def validate_or_repatch(
        state: ExtractionState,
    ) -> Literal["validate", "patch"]:
        if state["messages"][-1].type == "ai":
            return "validate"
        return "patch"

    builder.add_conditional_edges(
        "patch", validate_or_repatch, path_map=["validate", "patch", "__end__"]
    )
    compiled = builder.compile()
    compiled.name = "TrustCall"

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
            return ExtractionOutputs(
                messages=[], responses=[], attempts=state["attempts"]
            )
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
            "attempts": state["attempts"],
        }

    def coerce_inputs(state: InputsLike) -> Union[ExtractionInputs, dict]:
        """Coerce inputs to the expected format."""
        if isinstance(state, list):
            return {"messages": state}
        if isinstance(state, PromptValue):
            return {"messages": state.to_messages()}
        if isinstance(state, dict):
            if isinstance(state.get("messages"), PromptValue):
                state = {**state, "messages": state["messages"].to_messages()}  # type: ignore
        return cast(dict, state)

    return coerce_inputs | compiled | filter_state


def ensure_tools(
    tools: Sequence[TOOL_T],
) -> List[Union[BaseTool, Type[BaseModel], Callable]]:
    results: list = []
    for t in tools:
        if isinstance(t, dict):
            if all(k in t for k in ("name", "description", "parameters")):
                schema = create_model_from_schema(
                    {"title": t["name"], **t["parameters"]}
                )
                schema.__doc__ = (getattr(schema, __doc__, "") or "") + (
                    t.get("description") or ""
                )
                schema.__name__ = t["name"]
                results.append(schema)
            elif all(k in t for k in ("type", "function")):
                # Already in openai format
                resolved = ensure_tools([t["function"]])
                results.extend(resolved)
            else:
                model = create_model_from_schema(t)
                if not model.__doc__:
                    model.__doc__ = t.get("description") or model.__name__
                results.append(model)
        elif isinstance(t, (BaseTool, type)):
            results.append(t)
        elif callable(t):
            results.append(csff_(t))
        else:
            raise ValueError(f"Invalid tool type: {type(t)}")
    return list(results)


## Helper functions + reducers


def _exclude_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v if not isinstance(v, dict) else _exclude_none(v)
        for k, v in d.items()
        if v is not None
    }


def _get_schema(model: Type[BaseModel]) -> dict:
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()  # type: ignore
    return _exclude_none(schema)


class _Extract:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence,
        tool_choice: Optional[str] = None,
    ):
        self.bound_llm = llm.bind_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": getattr(t, "name", t.__name__),
                        "description": t.__doc__,
                        "parameters": _get_schema(t),
                    },
                }
                for t in tools
            ],
            tool_choice=tool_choice,
        )

    @ls.traceable
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
        return RunnableCallable(self.invoke, self.ainvoke, name="extract", trace=False)


class _ExtractUpdates:
    """Prompt an LLM to patch an existing schema.

    We have found this to be prefereable to re-generating
    the entire tool call from scratch in several ways:

    1. Fewer output tokens.
    2. Less likely to introduce new errors or drop important information.
    3. Easier for the LLM to generate.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[Mapping[str, Type[BaseModel]]] = None,
        enable_inserts: bool = False,
    ):
        new_tools: list = [PatchDoc]
        tool_choice = PatchDoc.__name__
        if enable_inserts:  # Also let the LLM know that we can extract NEW schemas.
            tools_ = [
                schema
                for name, schema in (tools or {}).items()
                if name not in {PatchDoc.__name__, PatchFunctionErrors.__name__}
            ]
            new_tools.extend(tools_)
            tool_choice = "any"
        self.enable_inserts = enable_inserts
        self.bound = llm.bind_tools(new_tools, tool_choice=tool_choice)
        self.tools = tools

    @ls.traceable
    def _setup(self, state: ExtractionState):
        messages = state["messages"]
        existing = state["existing"]
        if not existing:
            raise ValueError("No existing schemas provided.")
        schema_strings = []
        if isinstance(existing, dict):
            for k, v in existing.items():
                schema = self.tools.get(k) if self.tools else None
                if not schema:
                    schema_str = ""
                    logger.warning(
                        f"Schema {k} could not not be found for existing payload {v}"
                    )
                else:
                    schema_json = schema.schema()
                    schema_str = f"""
    <json_schema>
    {schema_json}
    </json_schema>
    """
                schema_strings.append(
                    f"<schema id={k}>\n<instance>\n{v}\n"
                    f"</instance>{schema_str}</schema>"
                )
        else:
            for schema_id, tname, d in existing:
                schema_strings.append(
                    f'<instance id={schema_id} schema_type="{tname}">\n{d}\n</instance>'
                )

        existing_schemas = "\n".join(schema_strings)
        cmd = "Generate JSONPatches to update the existing schema instances."
        if self.enable_inserts:
            cmd += (
                " If you need to extract or insert *new* instances of the schemas"
                ", call the relevant function(s)."
            )

        existing_msg = f"""{cmd}
<existing>
{existing_schemas}
</existing>
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

    @ls.traceable
    def _teardown(
        self,
        msg: AIMessage,
        existing: Union[Dict[str, Any], List[SchemaInstance]],
    ):
        resolved_tool_calls = []
        rt = ls.get_current_run_tree()
        for tc in msg.tool_calls:
            if tc["name"] == PatchDoc.__name__:
                json_doc_id = tc["args"]["json_doc_id"]
                if isinstance(existing, dict):
                    target = existing.get(str(json_doc_id))
                    tool_name = json_doc_id
                else:
                    try:
                        _, tool_name, target = next(
                            (e for e in existing if e[0] == json_doc_id),
                        )
                        if not tool_name:
                            raise ValueError("Could not find tool name")
                    except (ValueError, IndexError, TypeError):
                        logger.error(
                            f"Could not find existing schema in list for {json_doc_id}"
                        )
                        if rt:
                            rt.error = (
                                f"Could not find existing schema for {json_doc_id}"
                            )
                        continue

                if target:
                    try:
                        resolved_tool_calls.append(
                            ToolCall(
                                id=tc["id"],
                                name=tool_name,
                                args=jsonpatch.apply_patch(
                                    target, tc["args"]["patches"]
                                ),
                            )
                        )
                    except jsonpatch.JsonPatchConflict as e:
                        logger.error(f"Could not apply patch: {e}")
                        if rt:
                            rt.error = f"Could not apply patch: {e}"
                else:
                    if rt:
                        rt.error = f"Could not find existing schema for {tool_name}"
                    logger.warning(f"Could not find existing schema for {tool_name}")
            else:
                resolved_tool_calls.append(tc)
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

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Generate a JSONPatch to simply update an existing schema.

        Returns a single AIMessage with the updated schema, as if
            the schema were extracted from scratch.
        """
        messages, existing = self._setup(state)
        try:
            msg = await self.bound.ainvoke(messages, config)
            return self._teardown(cast(AIMessage, msg), existing)
        except Exception as e:
            return {
                "messages": [
                    HumanMessage(
                        content="Fix the validation error while"
                        f" also avoiding: {repr(str(e))}"
                    )
                ],
                "attempts": 1,
            }

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        messages, existing = self._setup(state)
        try:
            msg = self.bound.invoke(messages, config)
            return self._teardown(msg, existing)
        except Exception as e:
            return {
                "messages": [
                    HumanMessage(
                        content="Fix the validation error while"
                        f" also avoiding: {repr(str(e))}"
                    )
                ],
                "attempts": 1,
            }

    def as_runnable(self):
        return RunnableCallable(
            self.invoke, self.ainvoke, name="extract_updates", trace=False
        )


class _Patch:
    """Prompt an LLM to patch an invalid schema after it receives a ValidationError.

    We have found this to be more reliable and more token-efficient than
    re-creating the entire tool call from scratch.
    """

    def __init__(
        self, llm: BaseChatModel, valid_tool_names: Optional[List[str]] = None
    ):
        self.bound = llm.bind_tools(
            [PatchFunctionErrors, _create_patch_function_name_schema(valid_tool_names)],
            tool_choice="any",
        )

    @ls.traceable(tags=["patch", "langsmith:hidden"])
    def _tear_down(self, msg: AIMessage, messages: List[AnyMessage], target_id: str):
        if not msg.id:
            msg.id = str(uuid.uuid4())
        # We will directly update the messages in the state before validation.
        msg_ops = _infer_patch_message_ops(messages, msg.tool_calls, target_id)
        return {
            "messages": msg_ops,
            "attempts": 1,
        }

    async def ainvoke(
        self, state: ExtendedExtractState, config: RunnableConfig
    ) -> dict:
        """Generate a JSONPatch to correct the validation error and heal the tool call.

        Assumptions:
            - We only support a single tool call to be patched.
            - State's message list's last AIMessage contains the actual schema to fix.
            - The last ToolMessage contains the tool call to fix.

        """
        msg = await self.bound.ainvoke(state["messages"], config)
        return self._tear_down(msg, state["messages"], state["tool_call_id"])

    def invoke(self, state: ExtendedExtractState, config: RunnableConfig) -> dict:
        msg = self.bound.invoke(state["messages"], config)
        return self._tear_down(
            cast(AIMessage, msg), state["messages"], state["tool_call_id"]
        )

    def as_runnable(self):
        return RunnableCallable(self.invoke, self.ainvoke, name="patch", trace=False)


# We COULD just say Any for the value below, but Fireworks and some other
# providers don't support untyped arrays and dicts...
_JSON_PRIM_TYPES = Union[str, StrictInt, StrictBool, StrictFloat, None]
_JSON_TYPES = Union[
    _JSON_PRIM_TYPES, List[_JSON_PRIM_TYPES], Dict[str, _JSON_PRIM_TYPES]
]


class JsonPatch(BaseModel):
    """A JSON Patch document represents an operation to be performed on a JSON document.

    Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
    """  # noqa

    op: Literal["add", "remove", "replace"] = Field(
        ...,
        description="The operation to be performed. Must be one"
        " of 'add', 'remove', 'replace'.",
    )
    path: str = Field(
        ...,
        description="A JSON Pointer path that references a location within the"
        " target document where the operation is performed.",
    )
    value: Union[_JSON_TYPES, List[_JSON_TYPES], Dict[str, _JSON_TYPES]] = Field(
        ...,
        description="The value to be used within the operation. REQUIRED for"
        " 'add', 'replace', and 'test' operations."
        " Pay close attention to the json schema to ensure"
        " patched document will be valid.",
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "op": "add",
                    "path": "/path/to/my_array",
                    "patch_value": ["some", "values"],
                },
                {
                    "op": "add",
                    "path": "/path/to/my_array/1",
                    "patch_value": ["newer"],
                },
                {
                    "op": "replace",
                    "path": "/path/to/my_array/1",
                    "patch_value": "even newer",
                },
                {
                    "op": "remove",
                    "path": "/path/to/my_array/1",
                },
                {
                    "op": "replace",
                    "path": "/path/to/broken_object",
                    "patch_value": {"new": "object"},
                },
            ]
        }


# Used for fixing validation errors
class PatchFunctionErrors(BaseModel):
    """Respond with all JSONPatch operations required to update the previous invalid function call.

    Use to correct all validation errors in non-compliant function calls, or to extend or update existing structured data in the presence of new information. Closely analyze
    the parameters from the original JSONSchema to ensure the patched document will be valid
    and that you avoid repeating the same errors.
    """  # noqa

    json_doc_id: str = Field(
        ...,
        description="The ID of the function you are patching.",
    )
    planned_edits: str = Field(
        ...,
        description="Write a bullet-point list of each ValidationError you encountered"
        " and the corresponding JSONPatch operation needed to heal it."
        " For each operation, write why your initial guess was incorrect, "
        " citing the corresponding types(s) from the JSONSchema"
        " that will be used the validate the resultant patched document."
        "  Think step-by-step to ensure no error is overlooked.",
    )
    patches: list[JsonPatch] = Field(
        ...,
        description="A list of JSONPatch operations to be applied to the"
        " previous tool call's response arguments. If none are required, return"
        " an empty list. This field is REQUIRED.",
    )


def _create_patch_function_name_schema(valid_tool_names: Optional[List[str]] = None):
    if valid_tool_names:
        namestr = ", ".join(valid_tool_names)
        vname = f" Must be one of {namestr}"
    else:
        vname = ""

    class PatchFunctionName(BaseModel):
        """Call this if the tool message indicates that you previously invoked an invalid tool, (e.g., "Unrecognized tool name" error), do so here."""  # noqa

        json_doc_id: str = Field(
            ...,
            description="The ID of the function you are patching.",
        )
        reasoning: list[str] = Field(
            ...,
            description="At least 2 logical reasons why this action ought to be taken."
            "Cite the specific error(s) mentioned to motivate the fix.",
        )
        fixed_name: Optional[str] = Field(
            ...,
            description="If you need to change the name of the function (e.g., "
            f'from an "Unrecognized tool name" error), do so here.{vname}',
        )

    return PatchFunctionName


# Used for updating existing documents
class PatchDoc(BaseModel):
    """Respond with JSONPatch operations to update the existing JSON document based on the provided text and schema."""  # noqa

    json_doc_id: str = Field(
        ...,
        description="The json_doc_id of the document you are patching.",
    )
    planned_edits: str = Field(
        ...,
        description="Think step-by-step, reasoning over each required"
        " update and the corresponding JSONPatch operation to accomplish it."
        " Cite the fields in the JSONSchema you referenced in developing this plan.",
    )
    patches: list[JsonPatch] = Field(
        ...,
        description="A list of JSONPatch operations to be applied to the"
        " previous tool call's response. If none are required, return an empty list."
        " This field is REQUIRED.",
    )


class MessageOp(TypedDict):
    op: Literal["delete", "update_tool_call", "update_tool_name"]
    target: Union[str, ToolCall]


def _get_history_for_tool_call(messages: List[AnyMessage], tool_call_id: str):
    results = []
    seen_ai_message = False
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if not seen_ai_message:
                tool_calls = [tc for tc in m.tool_calls if tc["id"] == tool_call_id]
                if hasattr(m, "model_dump"):
                    d = m.model_dump(exclude={"tool_calls"})
                else:
                    d = m.dict(exclude={"tool_calls"})
                m = AIMessage(
                    **d,
                    tool_calls=tool_calls,
                )
            seen_ai_message = True
        if isinstance(m, ToolMessage):
            if m.tool_call_id != tool_call_id and not seen_ai_message:
                continue
        results.append(m)
    return list(reversed(results))


def _apply_message_ops(
    messages: Sequence[AnyMessage], message_ops: Sequence[MessageOp]
) -> List[AnyMessage]:
    # Apply operations to the messages
    messages = list(messages)
    for message_op in message_ops:
        if message_op["op"] == "delete":
            t = cast(str, message_op["target"])
            messages_ = [m for m in messages if cast(str, getattr(m, "id")) != t]
            messages = messages_
        elif message_op["op"] == "update_tool_call":
            targ = cast(ToolCall, message_op["target"])
            messages_ = []
            for m in messages:
                if isinstance(m, AIMessage):
                    old = m.tool_calls.copy()
                    new = [
                        targ if tc["id"] == targ["id"] else tc for tc in m.tool_calls
                    ]
                    if old != new:
                        m = m.copy()
                        m.tool_calls = new
                        if m.additional_kwargs.get("tool_calls"):
                            m.additional_kwargs["tool_calls"] = new
                    messages_.append(m)
                else:
                    messages_.append(m)
            messages = messages_
        elif message_op["op"] == "update_tool_name":
            targ = cast(dict, message_op["target"])
            messages_ = []
            for m in messages:
                if isinstance(m, AIMessage):
                    new = []
                    for tc in m.tool_calls:
                        if tc["id"] == targ["id"]:
                            new.append(
                                {
                                    "id": targ["id"],
                                    "name": targ["name"],  # Just updating the name
                                    "args": tc["args"],
                                }
                            )
                        else:
                            new.append(tc)
                    if m.tool_calls != new:
                        m = m.copy()
                        m.tool_calls = new
                    messages_.append(m)
            messages = messages_

        else:
            raise ValueError(f"Invalid operation: {message_op['op']}")
    return messages


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
    messages = cast(Sequence[AnyMessage], add_messages(left, right))  # type: ignore[arg-type]
    if message_ops:
        messages = _apply_message_ops(messages, message_ops)
    return messages


def _get_message_op(
    messages: Sequence[AnyMessage], tool_call: dict, tool_call_name: str, target_id: str
) -> List[MessageOp]:
    msg_ops: List[MessageOp] = []
    rt = ls.get_current_run_tree()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] == target_id:
                    if tool_call_name == "PatchFunctionName":
                        msg_ops.append(
                            {
                                "op": "update_tool_name",
                                "target": {
                                    "id": target_id,
                                    "name": tool_call["fixed_name"],
                                },
                            }
                        )
                    elif tool_call_name in ("PatchFunctionErrors", "PatchDoc"):
                        try:
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
                        except jsonpatch.JsonPatchConflict as e:
                            if rt:
                                rt.error = f"Could not apply patch: {repr(e)}"
                            logger.error(f"Could not apply patch: {repr(e)}")
                    else:
                        if rt:
                            rt.error = f"Unrecognized function call {tool_call_name}"
                        logger.error(f"Unrecognized function call {tool_call_name}")
        if isinstance(m, ToolMessage):
            if m.tool_call_id == target_id:
                msg_ops.append(MessageOp(op="delete", target=m.id or ""))
    return msg_ops


@ls.traceable(tags=["langsmith:hidden"])
def _infer_patch_message_ops(
    messages: Sequence[AnyMessage], tool_calls: List[ToolCall], target_id: str
):
    return [
        op
        for tool_call in tool_calls
        for op in _get_message_op(
            messages, tool_call["args"], tool_call["name"], target_id=target_id
        )
    ]


def csff_(function: Callable) -> Type[BaseModel]:
    fn = _strip_injected(function)
    schema = create_schema_from_function(function.__name__, fn)
    schema.__name__ = function.__name__
    return schema


def _keep_first(left: Any, right: Any):
    return left or right


class ExtractionState(TypedDict):
    messages: Annotated[List[AnyMessage], _reduce_messages]
    attempts: Annotated[int, operator.add]
    msg_id: Annotated[str, _keep_first]
    """Set once and never changed. The ID of the message to be patched."""
    existing: Optional[Dict[str, Any]]
    """If you're updating an existing schema, provide the existing schema here."""


class ExtendedExtractState(ExtractionState):
    tool_call_id: str
    """The ID of the tool call to be patched."""


class DeletionState(ExtractionState):
    deletion_target: str


class _ExtendedValidationNode(ValidationNode):
    def _func(
        self, input: Union[list[AnyMessage], dict[str, Any]], config: RunnableConfig
    ) -> Any:
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(input)

        def run_one(call: ToolCall):
            try:
                schema = self.schemas_by_name[call["name"]]
                output = schema.validate(call["args"])
                return ToolMessage(
                    content=output.json(),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except KeyError:
                valid_names = ", ".join(self.schemas_by_name.keys())
                return ToolMessage(
                    content=f"Unrecognized tool name: \"{call['name']}\". You only have"
                    f" access to the following tools: {valid_names}."
                    " Please call PatchFunctionName with the *correct* tool name"
                    f" to fix json_doc_id=[{call['id']}].",
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )
            except Exception as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}


def _is_injected_arg_type(type_: Type) -> bool:
    return any(
        isinstance(arg, InjectedToolArg)
        or (isinstance(arg, type) and issubclass(arg, InjectedToolArg))
        for arg in get_args(type_)[1:]
    )


def _curry(func: Callable, **fixed_kwargs: Any) -> Callable:
    """Bind parameters to a function, removing those parameters from the signature.

    Useful for exposing a narrower interface than what the the original function
    provides.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwargs = {**fixed_kwargs, **cast(fixed_kwargs, kwargs)}
        return func(*args, **new_kwargs)

    sig = inspect.signature(func)
    # Check that fixed_kwargs are all valid parameters of the function
    invalid_kwargs = set(fixed_kwargs) - set(sig.parameters)
    if invalid_kwargs:
        raise ValueError(f"Invalid parameters: {invalid_kwargs}")

    new_params = [p for name, p in sig.parameters.items() if name not in fixed_kwargs]
    wrapper.__signature__ = sig.replace(parameters=new_params)
    return wrapper


def _strip_injected(fn: Callable) -> Callable:
    """Strip injected arguments from a function's signature."""
    injected = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if _is_injected_arg_type(p.annotation)
    ]
    return _curry(fn, **{k: None for k in injected})


__all__ = [
    "create_extractor",
    "ensure_tools",
    "ExtractionInputs",
    "ExtractionOutputs",
]
