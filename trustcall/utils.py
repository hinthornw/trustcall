"""Utility functions for the trustcall package."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Type,
    get_args,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import InjectedToolArg

logger = logging.getLogger("extraction")


def is_gemini_model(llm: BaseChatModel) -> bool:
    """Determine if the provided LLM is a Google Vertex AI Gemini model."""
    # Check based on class module path
    if hasattr(llm, "__class__") and hasattr(llm.__class__, "__module__"):
        module_path = llm.__class__.__module__.lower()
        is_gemini_by_module = any(term in module_path for term in ["vertex", "google", "gemini"])
        if is_gemini_by_module:
            return True
    
    # Check based on model name, if available
    model_name = getattr(llm, "model_name", "") or ""
    is_gemini_by_name = isinstance(model_name, str) and "gemini" in model_name.lower()
    if is_gemini_by_name:
        return True
    
    return False


def _exclude_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary recursively."""
    return {
        k: v if not isinstance(v, dict) else _exclude_none(v)
        for k, v in d.items()
        if v is not None
    }



def _is_injected_arg_type(type_: Type) -> bool:
    """Check if a type is an injected argument type."""
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
        new_kwargs = {**fixed_kwargs, **kwargs}
        return func(*args, **new_kwargs)

    sig = inspect.signature(func)
    # Check that fixed_kwargs are all valid parameters of the function
    invalid_kwargs = set(fixed_kwargs) - set(sig.parameters)
    if invalid_kwargs:
        raise ValueError(f"Invalid parameters: {invalid_kwargs}")

    new_params = [p for name, p in sig.parameters.items() if name not in fixed_kwargs]
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore
    return wrapper


def _strip_injected(fn: Callable) -> Callable:
    """Strip injected arguments from a function's signature."""
    injected = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if _is_injected_arg_type(p.annotation)
    ]
    return _curry(fn, **{k: None for k in injected})


def _try_parse_json_value(value):
    """Try to parse a string value as JSON if it looks like JSON."""
    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _get_history_for_tool_call(messages: List[AnyMessage], tool_call_id: str):
    """Get the history of messages related to a specific tool call."""
    results = []
    seen_ai_message = False
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if not seen_ai_message:
                tool_calls = [tc for tc in m.tool_calls if tc["id"] == tool_call_id]
                if hasattr(m, "model_dump"):
                    d = m.model_dump(exclude={"tool_calls", "content"})
                else:
                    d = m.dict(exclude={"tool_calls", "content"})
                m = AIMessage(
                    **d,
                    # Frequently have partial_json blocks that are
                    # invalid if sent back to the API
                    content=str(m.content),
                    tool_calls=tool_calls,
                )
            seen_ai_message = True
        if isinstance(m, ToolMessage):
            if m.tool_call_id != tool_call_id and not seen_ai_message:
                continue
        results.append(m)
    return list(reversed(results))