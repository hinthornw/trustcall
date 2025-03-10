"""Extraction-related functionality for the trustcall package."""

from __future__ import annotations

import functools
import logging
import operator
import uuid
from dataclasses import asdict
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
import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.constants import Send
from langgraph.graph import StateGraph
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
from typing_extensions import TypedDict

from trustcall.patch import _Patch
from trustcall.schema import (
    _create_remove_doc_from_existing,
    _get_schema,
    _create_patch_function_errors_schema,
    _create_patch_doc_schema,
)
from trustcall.tools import TOOL_T, ensure_tools
from trustcall.types import (
    ExistingType,
    ExtractionInputs,
    ExtractionOutputs,
    InputsLike,
    Messages,
    SchemaInstance,
)
from trustcall.utils import is_gemini_model, _get_history_for_tool_call
from trustcall.validation import _ExtendedValidationNode
from trustcall.states import ExtractionState, ExtendedExtractState, DeletionState, MessageOp

logger = logging.getLogger("extraction")

DEFAULT_MAX_ATTEMPTS = 3

class _Extract:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence,
        tool_choice: Optional[str] = None,
        for_gemini: bool = False,
    ):
        # Create proper tool schemas based on the model type
        tool_schemas = []
        for t in tools:
            schema = _get_schema(t, for_gemini)
            tool_dict = {
                "type": "function",
                "function": {
                    "name": getattr(t, "name", t.__name__),
                    "description": t.__doc__,
                    "parameters": schema,
                }
            }
            tool_schemas.append(tool_dict)
            
        self.bound_llm = llm.bind_tools(tool_schemas, tool_choice=tool_choice)

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
        """Extract entities from the input messages."""
        msg = await self.bound_llm.ainvoke(state.messages, config)
        return self._tear_down(cast(AIMessage, msg))

    def invoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Extract entities from the input messages."""
        msg = self.bound_llm.invoke(state.messages, config)
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
        tools: Dict[str, Type[BaseModel]],
        enable_inserts: bool = False,
        enable_updates: bool = True,
        enable_deletes: bool = False,
        existing_schema_policy: bool | Literal["ignore"] = True,
    ):
        if not any((enable_inserts, enable_updates, enable_deletes)):
            raise ValueError(
                "At least one of enable_inserts, enable_updates,"
                " or enable_deletes must be True."
            )
        
        # Get the appropriate patching tools - Gemini supports simpler JSON schemas, so requires different tools
        using_gemini = is_gemini_model(llm)
        patch_doc = _create_patch_doc_schema(using_gemini)
        patch_function_errors = _create_patch_function_errors_schema(using_gemini)
        
        new_tools: list = [patch_doc] if enable_updates else []
        tool_choice = "PatchDoc" if not enable_deletes else "any"
        if enable_inserts:
            tools_ = [
                schema
                for name, schema in (tools or {}).items()
                if name not in {patch_doc.__name__, patch_function_errors.__name__}
            ]
            new_tools.extend(tools_)
            tool_choice = "any"
            
        self.enable_inserts = enable_inserts
        self.enable_updates = enable_updates
        self.bound_tools = new_tools
        self.tool_choice = tool_choice
        self.bound = llm.bind_tools(new_tools, tool_choice=tool_choice)
        self.enable_deletes = enable_deletes
        self.tools = dict(tools) | {schema_.__name__: schema_ for schema_ in new_tools}
        self.existing_schema_policy = existing_schema_policy
        self.using_gemini = using_gemini
        

    @ls.traceable(tags=["langsmith:hidden"])
    def _setup(self, state: ExtractionState):
        messages = state.messages
        existing = state.existing
        if not existing:
            raise ValueError("No existing schemas provided.")
        existing = self._validate_existing(existing)  # type: ignore[assignment]
        schema_strings = []
        if isinstance(existing, dict):
            for k, v in existing.items():
                if k not in self.tools and self.existing_schema_policy is False:
                    schema_str = "object"
                else:
                    schema = self.tools[k]
                    schema_json = _get_schema(schema, self.using_gemini)
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
        removal_schema = None
        if self.enable_deletes and existing:
            removal_schema = _create_remove_doc_from_existing(existing)
            bound_model = self.bound.bound.bind_tools(  # type: ignore
                self.bound_tools + [removal_schema],
                tool_choice=self.tool_choice,
            )
        else:
            bound_model = self.bound

        return [system_message] + messages, existing, removal_schema, bound_model

    @ls.traceable(tags=["langsmith:hidden"])
    def _teardown(
        self,
        msg: AIMessage,
        existing: Union[Dict[str, Any], List[Any]],
    ):
        resolved_tool_calls = []
        updated_docs = {}
        
         # Try to get trace ID from langfuse if available, otherwise continue without it
        try:
            from langfuse.decorators import langfuse_context
            rt = langfuse_context.get_current_trace_id()
        except (ImportError, AttributeError):
            # Langfuse not available, try langsmith
            try:
                rt = ls.get_current_run_tree()
            except (ImportError, AttributeError):
                # Neither available, continue without tracing
                pass

        for tc in msg.tool_calls:
            if tc["name"] == "PatchDoc":
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
                            raise ValueError(
                                "Could not find tool name "
                                f"for json_doc_id {json_doc_id}"
                            )
                    except StopIteration:
                        logger.error(
                            f"Could not find existing schema in dict for {json_doc_id}"
                        )
                        if rt:
                            rt.error = (
                                f"Could not find existing schema for {json_doc_id}"
                            )
                        continue
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
                        from trustcall.schema import _ensure_patches
                        patches = _ensure_patches(tc["args"])
                        if patches or self.tool_choice == "PatchDoc":
                            # The second condition is so that, when we are continuously
                            # updating a single doc, we will still include it in
                            # the output responses list; mainly for backwards
                            # compatibility
                            resolved_tool_calls.append(
                                ToolCall(
                                    id=tc["id"],
                                    name=tool_name,
                                    args=jsonpatch.apply_patch(target, patches),
                                )
                            )
                            updated_docs[tc["id"]] = str(json_doc_id)
                    except Exception as e:
                        logger.error(f"Could not apply patch: {e}")
                        if rt:
                            rt.error = f"Could not apply patch: {repr(e)}"
                else:
                    if rt:
                        rt.error = f"Could not find existing schema for {tool_name}"
                    logger.warning(f"Could not find existing schema for {tool_name}")
            else:
                resolved_tool_calls.append(tc)
        ai_message = AIMessage(
            content=msg.content,
            tool_calls=resolved_tool_calls,
            additional_kwargs={"updated_docs": updated_docs},
        )
        if not ai_message.id:
            ai_message.id = str(uuid.uuid4())

        return {
            "messages": [ai_message],
            "attempts": 1,
            "msg_id": ai_message.id,
        }

    @property
    def _provided_tools(self):
        return sorted(self.tools.keys() - {"PatchDoc", "PatchFunctionErrors"})

    def _validate_existing(
        self, existing: ExistingType
    ) -> Union[Dict[str, Any], List[Any]]:
        """Check that all existing schemas match a known schema or '__any__'."""
        if isinstance(existing, dict):
            # For each top-level key, see if it's recognized
            validated = {}
            for key, record in existing.items():
                if key in self.tools or key == "__any__":
                    validated[key] = record
                else:
                    # Key does not match known schema
                    if self.existing_schema_policy is True:
                        raise ValueError(
                            f"Key '{key}' doesn't match any schema. "
                            f"Known schemas: {list(self.tools.keys())}"
                        )
                    elif self.existing_schema_policy is False:
                        validated[key] = record
                    else:  # "ignore"
                        logger.warning(f"Ignoring unknown schema: {key}")
            return validated

        elif isinstance(existing, list):
            # For list types, validate each item's schema_name
            coerced = []
            for i, item in enumerate(existing):
                if hasattr(item, "record_id") and hasattr(item, "schema_name") and hasattr(item, "record"):
                    if (
                        item.schema_name not in self.tools
                        and item.schema_name != "__any__"
                    ):
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{item.schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(
                                    item.record_id, item.schema_name, item.record
                                )
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema at index {i}")
                            continue
                    else:
                        coerced.append(item)
                elif isinstance(item, tuple) and len(item) == 3:
                    record_id, schema_name, record_dict = item
                    if isinstance(record_dict, BaseModel):
                        record_dict = record_dict.model_dump(mode="json")
                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(record_id, schema_name, record_dict)
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        coerced.append(
                            SchemaInstance(record_id, schema_name, record_dict)
                        )
                elif isinstance(item, tuple) and len(item) == 2:
                    # Assume record_ID, item
                    record_id, model = item
                    if hasattr(model, "__name__"):
                        schema_name = model.__name__
                    else:
                        schema_name = model.__repr_name__()

                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            val = (
                                model.model_dump(mode="json")
                                if isinstance(model, BaseModel)
                                else model
                            )
                            coerced.append(SchemaInstance(record_id, schema_name, val))
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        val = (
                            model.model_dump(mode="json")
                            if isinstance(model, BaseModel)
                            else model
                        )
                        coerced.append(SchemaInstance(record_id, schema_name, val))
                elif isinstance(item, BaseModel):
                    if hasattr(item, "__name__"):
                        schema_name = item.__name__
                    else:
                        schema_name = item.__repr_name__()

                    if schema_name not in self.tools and schema_name != "__any__":
                        if self.existing_schema_policy is True:
                            raise ValueError(
                                f"Unknown schema '{schema_name}' at index {i}"
                            )
                        elif self.existing_schema_policy is False:
                            coerced.append(
                                SchemaInstance(
                                    str(uuid.uuid4()),
                                    schema_name,
                                    item.model_dump(mode="json"),
                                )
                            )
                        else:  # "ignore"
                            logger.warning(f"Ignoring unknown schema '{schema_name}'")
                            continue
                    else:
                        coerced.append(
                            SchemaInstance(
                                str(uuid.uuid4()),
                                schema_name,
                                item.model_dump(mode="json"),
                            )
                        )
                else:
                    raise ValueError(
                        f"Invalid item at index {i} in existing list."
                        f" Provided: {item}, Expected: SchemaInstance"
                        f" or Tuple[str, str, dict] or BaseModel"
                    )
            return coerced
        else:
            raise ValueError(
                f"Invalid type for existing. Provided: {type(existing)},"
                f" Expected: dict or list. Supported formats are:\n"
                "1. Dict[str, Any] where keys are tool names\n"
                "2. List[SchemaInstance]\n3. List[Tuple[str, str, Dict[str, Any]]]"
            )

    async def ainvoke(self, state: ExtractionState, config: RunnableConfig) -> dict:
        """Generate a JSONPatch to simply update an existing schema.

        Returns a single AIMessage with the updated schema, as if
            the schema were extracted from scratch.
        """
        messages, existing, removal_schema, bound_model = self._setup(state)
        try:
            msg = await bound_model.ainvoke(messages, config)
            return {
                **self._teardown(cast(AIMessage, msg), existing),
                "removal_schema": removal_schema,
            }
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
        messages, existing, removal_schema, bound_model = self._setup(state)
        try:
            msg = bound_model.invoke(messages, config)
            return {**self._teardown(msg, existing), "removal_schema": removal_schema}
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


def create_extractor(
    llm: str | BaseChatModel,
    *,
    tools: Sequence[TOOL_T],
    tool_choice: Optional[str] = None,
    enable_inserts: bool = False,
    enable_updates: bool = True,
    enable_deletes: bool = False,
    existing_schema_policy: bool | Literal["ignore"] = True,
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
        enable_updates (bool): Whether to allow the LLM to update existing schemas
            using the PatchDoc tool. (default: True)
        enable_deletes (bool): Whether to allow the LLM to delete existing schemas
            using the RemoveDoc tool. (default: False)
        existing_schema_policy (bool | Literal["ignore"]): How to handle existing schemas
            that don't match the provided tool. Useful for migrating or managing heterogenous
            docs. (default: True) True means raise error. False means treat as dict.
            "ignore" means ignore (drop any attempts to patch these)

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
    # Convert string to model if needed
    if isinstance(llm, str):
        try:
            from langchain.chat_models import init_chat_model
            llm = init_chat_model(llm)
        except ImportError:
            raise ImportError(
                "Creating extractors from a string requires langchain>=0.3.0,"
                " as well as the provider-specific package"
                " (like langchain-openai, langchain-anthropic, etc.)"
                " Please install langchain to continue."
            )
    builder = StateGraph(ExtractionState)
    
    # Check if the model is a Gemini model - this affects the schema generation and patching
    using_gemini = is_gemini_model(llm)

    # Define error formatting
    def format_exception(error: BaseException, call: ToolCall, schema: Type[BaseModel]) -> str:
        return (
            f"Error:\n\n```\n{str(error)}\n```\n"
            "Expected Parameter Schema:\n\n" + f"```json\n{_get_schema(schema, using_gemini)}\n```\n"
            f"Please use PatchFunctionErrors to fix all validation errors."
            f" for json_doc_id=[{call['id']}]."
        )
    
    # Get the appropriate patching tools - Gemini supports simpler JSON schemas, so requires different tools
    patch_doc = _create_patch_doc_schema(using_gemini)
    patch_function_errors = _create_patch_function_errors_schema(using_gemini)

    # Create validator with appropriate tools
    validator = _ExtendedValidationNode(
        ensure_tools(tools) + [patch_doc, patch_function_errors],
        format_error=format_exception,  # type: ignore
        enable_deletes=enable_deletes,
    )
    _extract_tools = [
        schema
        for name, schema in validator.schemas_by_name.items()
        if name not in {patch_doc.__name__, patch_function_errors.__name__}
    ]
    tool_names = [getattr(t, "name", t.__name__) for t in _extract_tools]
    builder.add_node(
        _Extract(
            llm,
            _extract_tools,
            tool_choice,
            for_gemini=using_gemini,
        ).as_runnable()
    )
    updater = _ExtractUpdates(
        llm,
        tools=validator.schemas_by_name.copy(),
        enable_inserts=enable_inserts,  # type: ignore
        enable_updates=enable_updates,  # type: ignore
        enable_deletes=enable_deletes,  # type: ignore
        existing_schema_policy=existing_schema_policy,
    )
    builder.add_node(updater.as_runnable())
    builder.add_node(_Patch(llm, valid_tool_names=tool_names).as_runnable())
    builder.add_node("validate", validator)

    def del_tool_call(state: DeletionState) -> dict:
        return {
            "messages": MessageOp(op="delete", target=state.deletion_target),
        }

    builder.add_node(del_tool_call)

    def enter(state: ExtractionState) -> Literal["extract", "extract_updates"]:
        if state.existing:
            return "extract_updates"
        return "extract"

    builder.add_conditional_edges("__start__", enter)

    def validate_or_retry(
        state: ExtractionState,
    ) -> Literal["validate", "extract_updates"]:
        if state.messages[-1].type == "ai":
            return "validate"
        return "extract_updates"

    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("extract_updates", validate_or_retry)

    def handle_retries(state: ExtractionState, config: RunnableConfig) -> Union[Literal["__end__"], list]:
        """After validation, decide whether to retry or end the process."""
        max_attempts = config["configurable"].get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        if state.attempts >= max_attempts:
            return "__end__"
        # Only continue if we need to patch the tool call
        to_send = []
        bumped = False
        
        # Add defensive check - ensure there's at least one AIMessage in history
        has_ai_message = any(isinstance(m, AIMessage) for m in state.messages)
        if not has_ai_message:
            logger.warning("No AIMessage found in state.messages, ending processing")
            return "__end__"
            
        for m in reversed(state.messages):
            if isinstance(m, AIMessage):
                break
            if isinstance(m, ToolMessage):
                if m.status == "error":
                    messages_for_fixing = _get_history_for_tool_call(
                        state.messages, m.tool_call_id
                    )
                    
                    # Ensure tool_call_id is properly set
                    if not hasattr(m, "tool_call_id") or not m.tool_call_id:
                        logger.warning(f"Missing tool_call_id on message {m}, skipping")
                        continue
                        
                    to_send.append(
                        Send(
                            "patch",
                            ExtendedExtractState(
                                **{
                                    **asdict(state),
                                    "messages": messages_for_fixing,
                                    "tool_call_id": m.tool_call_id,
                                    "bump_attempt": not bumped,
                                }
                            ),
                        )
                    )
                    bumped = True
                else:
                    # Safe deletion handling
                    if not hasattr(m, "id") or not m.id:
                        logger.warning(f"Missing id on message {m}, skipping deletion")
                        continue
                    # We want to delete the validation tool calls
                    # anyway to avoid mixing branches during fan-in
                    to_send.append(
                        Send(
                            "del_tool_call",
                            DeletionState(
                                deletion_target=str(m.id), messages=state.messages
                            ),
                        )
                    )
        return to_send

    builder.add_conditional_edges(
        "validate", handle_retries, path_map=["__end__", "patch", "del_tool_call"]
    )

    def sync(state: ExtractionState, config: RunnableConfig) -> dict:
        return {"messages": []}

    def validate_or_repatch(
        state: ExtractionState,
    ) -> Literal["validate", "patch"]:
        if state.messages[-1].type == "ai":
            return "validate"
        return "patch"

    builder.add_node(sync)

    builder.add_conditional_edges(
        "sync", validate_or_repatch, path_map=["validate", "patch", "__end__"]
    )
    compiled = builder.compile(checkpointer=False)
    compiled.name = "TrustCall"

    def filter_state(state: dict) -> ExtractionOutputs:
        """Filter the state to only include the validated AIMessage + responses."""
        msg_id = state["msg_id"]
        msg: Optional[AIMessage] = next(
            (
                m
                for m in state["messages"]
                if m.id == msg_id and isinstance(m, AIMessage)
            ),  # type: ignore
            None,
        )
        if not msg:
            return ExtractionOutputs(
                messages=[],
                responses=[],
                attempts=state["attempts"],
                response_metadata=[],
            )
        responses = []
        response_metadata = []
        updated_docs = msg.additional_kwargs.get("updated_docs") or {}
        existing = state.get("existing")
        removal_schema = None
        if enable_deletes and existing:
            removal_schema = _create_remove_doc_from_existing(existing)
        for tc in msg.tool_calls:
            if removal_schema and tc["name"] == removal_schema.__name__:
                sch = removal_schema
            elif tc["name"] not in validator.schemas_by_name:
                if existing_schema_policy in (False, "ignore"):
                    continue
                sch = validator.schemas_by_name[tc["name"]]
            else:
                sch = validator.schemas_by_name[tc["name"]]
            try:
                responses.append(
                    sch.model_validate(tc["args"])
                    if hasattr(sch, "model_validate")
                    else sch.parse_obj(tc["args"])
                )
                meta = {"id": tc["id"]}
                if json_doc_id := updated_docs.get(tc["id"]):
                    meta["json_doc_id"] = json_doc_id
                response_metadata.append(meta)
            except Exception as e:
                logger.error(e)
                continue

        return {
            "messages": [msg],
            "responses": responses,
            "response_metadata": response_metadata,
            "attempts": state["attempts"],
        }

    def coerce_inputs(state: InputsLike) -> Union[ExtractionInputs, dict]:
        """Coerce inputs to the expected format."""
        if isinstance(state, list):
            return {"messages": state}
        if isinstance(state, str):
            return {"messages": [{"role": "user", "content": state}]}
        if isinstance(state, PromptValue):
            return {"messages": state.to_messages()}
        if isinstance(state, dict):
            if isinstance(state.get("messages"), PromptValue):
                state = {**state, "messages": state["messages"].to_messages()}  # type: ignore
        else:
            if hasattr(state, "messages"):
                state = {"messages": state.messages.to_messages()}  # type: ignore

        return cast(dict, state)

    return coerce_inputs | compiled | filter_state