"""Patching-related functionality for the trustcall package."""

from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Sequence,
    Union,
    Optional,
    cast,
    
)

import jsonpatch  # type: ignore[import-untyped]
import langsmith as ls
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable

from trustcall.schema import _ensure_patches, _create_patch_function_errors_schema, _create_patch_function_name_schema
from trustcall.states import ExtendedExtractState, MessageOp
from trustcall.utils import is_gemini_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("extraction")


class _Patch:
    """Prompt an LLM to patch an invalid schema after it receives a ValidationError.

    We have found this to be more reliable and more token-efficient than
    re-creating the entire tool call from scratch.
    """

    def __init__(
        self, llm: BaseChatModel, valid_tool_names: Optional[List[str]] = None
    ):
        # Get the appropriate patching tools based on LLM type
        using_gemini = is_gemini_model(llm)
        self.bound = llm.bind_tools(
            [
                _create_patch_function_errors_schema(using_gemini), 
                _create_patch_function_name_schema(valid_tool_names, using_gemini)
                ],
            tool_choice="any",
        )

    @ls.traceable(tags=["patch", "langsmith:hidden"])
    def _tear_down(
        self,
        msg: AIMessage,
        messages: List[AnyMessage],
        target_id: str,
        bump_attempt: bool,
    ):
        if not msg.id:
            msg.id = str(uuid.uuid4())
        # We will directly update the messages in the state before validation.
        msg_ops = _infer_patch_message_ops(messages, msg.tool_calls, target_id)
        return {
            "messages": msg_ops,
            "attempts": 1 if bump_attempt else 0,
        }

    async def ainvoke(
        self, state: ExtendedExtractState, config: RunnableConfig
    ) -> Command[Literal["sync", "__end__"]]:
        """Generate a JSONPatch to correct the validation error and heal the tool call.

        Assumptions:
            - We only support a single tool call to be patched.
            - State's message list's last AIMessage contains the actual schema to fix.
            - The last ToolMessage contains the tool call to fix.

        """
        try:
            msg = await self.bound.ainvoke(state.messages, config)
        except Exception:
            return Command(goto="__end__")
        return Command(
            update=self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                state.tool_call_id,
                state.bump_attempt,
            ),
            goto=("sync",),
        )

    def invoke(
        self, state: ExtendedExtractState, config: RunnableConfig
    ) -> Command[Literal["sync", "__end__"]]:
        try:
            msg = self.bound.invoke(state.messages, config)
        except Exception:
            return Command(goto="__end__")
        return Command(
            update=self._tear_down(
                cast(AIMessage, msg),
                state.messages,
                state.tool_call_id,
                state.bump_attempt,
            ),
            goto=("sync",),
        )

    def as_runnable(self):
        return RunnableCallable(self.invoke, self.ainvoke, name="patch", trace=False)


def _get_message_op(
    messages: Sequence[AnyMessage], tool_call: dict, tool_call_name: str, target_id: str
) -> List[MessageOp]:
    msg_ops: List[MessageOp] = []
    
    # Process each message
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] == target_id:
                    # Handle PatchFunctionName
                    if tool_call_name == "PatchFunctionName":
                        if not tool_call.get("fixed_name"):
                            continue
                        msg_ops.append({
                            "op": "update_tool_name",
                            "target": {
                                "id": target_id,
                                "name": str(tool_call["fixed_name"]),
                            },
                        })
                    # Handle any patch function - cover all cases using name check instead of type check
                    elif "PatchFunctionErrors" in tool_call_name or tool_call_name == "PatchDoc":
                        try:
                            patches = _ensure_patches(tool_call)
                            if patches:
                                patched_args = jsonpatch.apply_patch(tc["args"], patches)
                                msg_ops.append({
                                    "op": "update_tool_call",
                                    "target": {
                                        "id": target_id,
                                        "name": tc["name"],
                                        "args": patched_args,
                                    },
                                })
                        except Exception as e:
                            logger.error(f"Could not apply patch: {repr(e)}")
                    else:
                        logger.error(f"Unrecognized function call {tool_call_name}")
        
        # Add delete operations for tool messages
        if isinstance(m, ToolMessage) and m.tool_call_id == target_id:
            msg_ops.append(MessageOp(op="delete", target=m.id or ""))
    
    return msg_ops


@ls.traceable(tags=["langsmith:hidden"])
def _infer_patch_message_ops(
    messages: Sequence[AnyMessage], tool_calls: List[ToolCall], target_id: str
):
    ops = [
        op
        for tool_call in tool_calls
        for op in _get_message_op(
            messages, tool_call["args"], tool_call["name"], target_id=target_id
        )
    ]
    return ops