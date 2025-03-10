"""Tool-related functionality for the trustcall package."""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Type,
    Union,
    cast,
    get_args,
)

from dydantic import create_model_from_schema
from langchain_core.tools import BaseTool, create_schema_from_function
from pydantic import BaseModel
from typing_extensions import get_origin, is_typeddict, Annotated

from trustcall.utils import _strip_injected

TOOL_T = Union[BaseTool, Type[BaseModel], Callable, Dict[str, Any]]
"""Type for tools that can be used with the extractor.

Can be one of:
- BaseTool: A LangChain tool
- Type[BaseModel]: A Pydantic model class
- Callable: A function
- Dict[str, Any]: A dictionary representing a schema
"""


def ensure_tools(
    tools: Sequence[TOOL_T],
) -> List[Union[BaseTool, Type[BaseModel], Callable]]:
    """Convert various tool formats to a consistent format.
    
    Args:
        tools: A sequence of tools in various formats
        
    Returns:
        A list of tools in a consistent format
        
    Raises:
        ValueError: If a tool is in an invalid format
    """
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
        elif is_typeddict(t):
            results.append(_convert_any_typed_dicts_to_pydantic(cast(type, t)))
        elif isinstance(t, (BaseTool, type)):
            results.append(t)
        elif callable(t):
            results.append(csff_(t))
        else:
            raise ValueError(f"Invalid tool type: {type(t)}")
    return list(results)


def csff_(function: Callable) -> Type[BaseModel]:
    """Create a schema from a function.
    
    Args:
        function: The function to create a schema from
        
    Returns:
        A Pydantic model class representing the function's schema
    """
    fn = _strip_injected(function)
    schema = create_schema_from_function(function.__name__, fn)
    schema.__name__ = function.__name__
    return schema


_MAX_TYPED_DICT_RECURSION = 25


def _convert_any_typed_dicts_to_pydantic(
    type_: type,
    *,
    visited: dict | None = None,
    depth: int = 0,
) -> type:
    """Convert TypedDict to Pydantic model.
    
    Args:
        type_: The type to convert
        visited: A dictionary of already visited types
        depth: The current recursion depth
        
    Returns:
        The converted type
    """
    from pydantic import Field, create_model
    
    visited = visited if visited is not None else {}
    if type_ in visited:
        return visited[type_]
    elif depth >= _MAX_TYPED_DICT_RECURSION:
        return type_
    elif is_typeddict(type_):
        typed_dict = type_
        docstring = inspect.getdoc(typed_dict)
        annotations_ = typed_dict.__annotations__
        fields: dict = {}
        for arg, arg_type in annotations_.items():
            if get_origin(arg_type) is Annotated:
                annotated_args = get_args(arg_type)
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    annotated_args[0], depth=depth + 1, visited=visited
                )
                field_kwargs = dict(zip(("default", "description"), annotated_args[1:]))
                if (field_desc := field_kwargs.get("description")) and not isinstance(
                    field_desc, str
                ):
                    raise ValueError(
                        f"Invalid annotation for field {arg}. Third argument to "
                        f"Annotated must be a string description, received value of "
                        f"type {type(field_desc)}."
                    )
                else:
                    pass
                fields[arg] = (new_arg_type, Field(**field_kwargs))
            else:
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    arg_type, depth=depth + 1, visited=visited
                )
                field_kwargs = {"default": ...}
                fields[arg] = (new_arg_type, Field(**field_kwargs))
        model = create_model(typed_dict.__name__, **fields)
        model.__doc__ = docstring or ""
        visited[typed_dict] = model
        return model
    elif (origin := get_origin(type_)) and (type_args := get_args(type_)):
        type_args = tuple(
            _convert_any_typed_dicts_to_pydantic(arg, depth=depth + 1, visited=visited)
            for arg in type_args  # type: ignore[index]
        )
        return origin[type_args]  # type: ignore[index]
    else:
        return type_