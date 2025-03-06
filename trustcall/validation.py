"""Validation-related functionality for the trustcall package."""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_validator import ValidationNode, get_executor_for_config
from dataclasses import asdict

logger = logging.getLogger("extraction")


class _ExtendedValidationNode(ValidationNode):
    """Extended validation node with support for deletion."""
    
    def __init__(self, *args, enable_deletes: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_deletes = enable_deletes

    def _func(self, input: Any, config: RunnableConfig) -> Any:  # type: ignore
        """Validate and run tool calls synchronously."""
        output_type, message = self._get_message(asdict(input))
        removal_schema = None
        if self.enable_deletes and hasattr(input, "existing") and input.existing:
            from trustcall.schema import _create_remove_doc_from_existing
            removal_schema = _create_remove_doc_from_existing(input.existing)
            
        # ADDED: Get the current attempt count from the state
        attempt_count = input.attempts if hasattr(input, 'attempts') else 1

        def run_one(call: ToolCall): # type: ignore
            try:
                if removal_schema and call["name"] == removal_schema.__name__:
                    schema = removal_schema
                else:
                    schema = self.schemas_by_name[call["name"]]
                try:
                    # ADDED: Create validation context with attempt count
                    validation_context = {"attempt_count": attempt_count}
                    # MODIFIED: Pass context to model_validate
                    output = schema.model_validate(call["args"], context=validation_context)
                    return ToolMessage(
                        content=output.model_dump_json(),
                        name=call["name"],
                        tool_call_id=cast(str, call["id"]),
                    )
                except Exception as validation_error:
                    raise validation_error

            except KeyError:
                valid_names = ", ".join(self.schemas_by_name.keys())
                return ToolMessage(
                    content=f'Unrecognized tool name: "{call["name"]}". You only have'
                    f" access to the following tools: {valid_names}."
                    " Please call PatchFunctionName with the *correct* tool name"
                    f" to fix json_doc_id=[{call['id']}].",
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    status="error",
                )
            except Exception as e:
                error_message = self._format_error(e, call, schema)
                return ToolMessage(
                    content=error_message,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    status="error",
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}