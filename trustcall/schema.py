"""
Handles the creation, conversion, and management of schemas used for tool calling,
validation, and patching, ensuring that trustcall can work with different 
LLMs (including Gemini) and various schema formats.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)

from trustcall.utils import _exclude_none

logger = logging.getLogger("extraction")


def create_gemini_compatible_schema(model_class):
    """
    Create a Gemini-compatible schema from a Pydantic model.
    
    Args:
        model_class: The Pydantic model class
        
    Returns:
        A Gemini-compatible schema dictionary
    """
    # Start with basic model info
    gemini_schema = {
        "type": "OBJECT",
        "title": model_class.__name__,
        "description": model_class.__doc__ or f"A {model_class.__name__} object",
        "properties": {},
        "required": []
    }
    
    # Get the field names in the order they were defined
    # This is crucial for Gemini's expected property ordering
    field_names = list(model_class.model_fields.keys())
    
    # Process all model fields
    for field_name in field_names:
        field = model_class.model_fields[field_name]
        
        # Add to required list if appropriate
        if field.is_required():
            gemini_schema["required"].append(field_name)
            
        # Get field description
        field_desc = field.description or f"The {field_name} field"
        
        # Convert field type to Gemini format
        gemini_schema["properties"][field_name] = convert_field_to_gemini(field, field_desc)
    
    return gemini_schema


def convert_field_to_gemini(field, description):
    """Convert a Pydantic field to Gemini-compatible schema format."""
    annotation = field.annotation
    
    # Handle basic types
    if annotation is str:
        return {"type": "STRING", "description": description}
    elif annotation is int:
        return {"type": "INTEGER", "description": description}
    elif annotation is float:
        return {"type": "NUMBER", "description": description}
    elif annotation is bool:
        return {"type": "BOOLEAN", "description": description}
    
    # Handle container types
    origin = get_origin(annotation)
    if origin is list:
        item_type = get_args(annotation)[0]
        return {
            "type": "ARRAY", 
            "description": description,
            "items": convert_type_to_gemini(item_type)
        }
    elif origin is dict:
        return {"type": "OBJECT", "description": description}
    elif origin is Union or origin is Optional:
        # For Union/Optional types, use the first type as primary
        # and add nullable if None is an option
        types = get_args(annotation)
        primary_type = next((t for t in types if t is not type(None)), types[0])
        result = convert_type_to_gemini(primary_type)
        if type(None) in types:
            # Unlike JSON Schema which uses nullable, Gemini might need a different approach
            # Just adding nullable true seems most reasonable
            result["nullable"] = True
        return result
    
    # Handle nested Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        # For nested models, recursively generate the schema
        # Gemini may not support references, so include the full schema
        return create_gemini_compatible_schema(annotation)
    
    # Default to string for unknown types
    return {"type": "STRING", "description": description}


def convert_type_to_gemini(type_annotation):
    """Convert a Python type to a Gemini schema type definition."""
    if type_annotation is str:
        return {"type": "STRING"}
    elif type_annotation is int:
        return {"type": "INTEGER"}
    elif type_annotation is float:
        return {"type": "NUMBER"}
    elif type_annotation is bool:
        return {"type": "BOOLEAN"}
    elif isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
        return create_gemini_compatible_schema(type_annotation)
    
    # Default to string
    return {"type": "STRING"}


def _get_schema(model: Type[BaseModel], for_gemini: bool) -> dict:    
    if for_gemini:
        return create_gemini_compatible_schema(model)
    else:
        if hasattr(model, "model_json_schema"):
            schema = model.model_json_schema()
        else:
            schema = model.schema()  # type: ignore
        return _exclude_none(schema)


# JSON Patch related classes

# We COULD just say Any for the value below, but Fireworks and some other
# providers don't support untyped arrays and dicts...
_JSON_PRIM_TYPES = Union[str, StrictInt, StrictBool, StrictFloat, None]
_JSON_TYPES = Union[
    _JSON_PRIM_TYPES, List[_JSON_PRIM_TYPES], Dict[str, _JSON_PRIM_TYPES]
]


class BasePatch(BaseModel):
    """Base class for all patch types."""
    op: Literal["add", "remove", "replace"] = Field(
        ...,
        description="A JSON Pointer path that references a location within the"
        " target document where the operation is performed."
        " Note: patches are applied sequentially. If you remove a value, the collection"
        " size changes before the next patch is applied.",
    )
    path: str = Field(
        ...,
        description="A JSON Pointer path that references a location within the"
        " target document where the operation is performed."
        " Note: patches are applied sequentially. If you remove a value, the collection"
        " size changes before the next patch is applied.",
    )


class FullPatch(BasePatch):
    """A JSON Patch document represents an operation to be performed on a JSON document.

    Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
    This supports OpenAI and other LLMs with full JSON support (not Gemini).
    """ # noqa
    value: Union[_JSON_TYPES, List[_JSON_TYPES], Dict[str, _JSON_TYPES]] = Field(
        ...,
        description="The value to be used within the operation."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "op": "replace",
                    "path": "/path/to/my_array/1",
                    "value": "the newer value to be patched",
                },
                {
                    "op": "replace",
                    "path": "/path/to/broken_object",
                    "value": {"new": "object"},
                },
                {
                    "op": "add",
                    "path": "/path/to/my_array/-",
                    "value": ["some", "values"],
                },
                {
                    "op": "add",
                    "path": "/path/to/my_array/-",
                    "value": ["newer"],
                },
                {
                    "op": "remove",
                    "path": "/path/to/my_array/1",
                },
            ]
        }
    )

class GeminiJsonPatch(BasePatch):
    """A JSON Patch document represents an operation to be performed on a JSON document.

    Note that the op and path are ALWAYS required. Value is required for ALL operations except 'remove'.
    This supports Gemini with it's more limited JSON compatibility.
    """ # noqa
    # Similar to JsonPatch but with Gemini-compatible schema definition
    # Instead of using a string-only value, use Union types that match Gemini's schema
    value: Optional[Union[str, int, float, bool, List, Dict]] = Field(
        default=None,
        description="The value to be used within the operation. Required for"
        " 'add' and 'replace' operations, not needed for 'remove'."
    )
    
    # For Gemini, we'll use a string value but with clear documentation that it can be complex
    value: Optional[str] = Field(
        default=None,
        description="The value to be used within the operation. For complex values (objects, arrays), "
        "provide valid JSON as a string. Required for 'add' and 'replace' operations."
    )
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v, info):
        """Automatically convert complex values to JSON strings and handle remove operations."""
        values = info.data
        
        # Allow None for remove operations
        if v is None and values.get("op") == "remove":
            return v
            
        # Convert objects and arrays to JSON strings
        if isinstance(v, (dict, list)):
            return json.dumps(v)
            
        # Convert primitive types to strings
        if v is not None and not isinstance(v, str):
            return str(v)
            
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "type": "OBJECT",
            "properties": {
                "op": {
                    "type": "STRING",
                    "enum": ["add", "remove", "replace"],
                    "description": "The operation to be performed."
                },
                "path": {
                    "type": "STRING",
                    "description": "JSON Pointer path where the operation is performed."
                },
                "value": {
                    "type": "STRING",
                    "description": "Value to use in the operation. For complex values, use JSON strings."
                }
            },
            "required": ["op", "path"]
        }
    )

def get_patch_class(for_gemini: bool) -> Type[BasePatch]:
    """Return the appropriate patch class based on the LLM type."""
    return GeminiJsonPatch if for_gemini else FullPatch

def _create_patch_function_errors_schema(for_gemini: bool = False) -> Type[BaseModel]:
    """Create the appropriate PatchFunctionErrors model based on the LLM type."""
    # Choose the appropriate patch type
    patch_class = get_patch_class(for_gemini)
    
    class PatchFunctionErrors(BaseModel):
        """Respond with all JSONPatch operations required to update the previous invalid function call."""
        
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
        patches: list[patch_class] = Field(
            ...,
            description="A list of JSONPatch operations to be applied to the"
            " previous tool call's response arguments. If none are required, return"
            " an empty list. This field is REQUIRED."
            " Multiple patches in the list are applied sequentially in the order provided,"
            " with each patch building upon the result of the previous one.",
        )

    return PatchFunctionErrors

def _create_patch_doc_schema(for_gemini: bool = False) -> Type[BaseModel]:
    """Create the appropriate PatchDoc model based on the LLM type."""
    
    patch_class = get_patch_class(for_gemini)

    
    class PatchDoc(BaseModel):
        """Respond with JSONPatch operations to update the existing JSON document based on the provided text and schema."""
        
        json_doc_id: str = Field(
            ...,
            description="The json_doc_id of the document you are patching.",
        )
        planned_edits: str = Field(
            ...,
            description="Think step-by-step, reasoning over each required"
            " update and the corresponding JSONPatch operation to accomplish it."
            " Cite the fields in the JSONSchema you referenced in developing this plan."
            " Address each path as a group; don't switch between paths.\n"
            " Plan your patches in the following order:"
            "1. replace - this keeps collection size the same.\n"
            "2. remove - BE CAREFUL ABOUT ORDER OF OPERATIONS."
            " Each operation is applied sequentially."
            " For arrays, remove the highest indexed value first to avoid shifting"
            " indices. This ensures subsequent remove operations remain valid.\n"
            " 3. add (for arrays, use /- to efficiently append to end).",
        )
        # For Gemini, we use a list of simple dictionaries instead of complex models
        patches: List[patch_class] = Field(
            ...,
            description="A list of JSONPatch operations to be applied to the"
            " previous tool call's response arguments. If none are required, return"
            " an empty list. This field is REQUIRED."
            " Multiple patches in the list are applied sequentially in the order provided,"
            " with each patch building upon the result of the previous one."
            " Take care to respect array bounds. Order patches as follows:\n"
            " 1. replace - this keeps collection size the same\n"
            " 2. remove - BE CAREFUL about order of operations. For arrays, remove"
            " the highest indexed value first to avoid shifting indices.\n"
            " 3. add - for arrays, use /- to efficiently append to end.",
        )
    
    return PatchDoc

def _create_patch_function_name_schema(valid_tool_names: Optional[List[str]] = None, for_gemini: bool = False):
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

        # If using Gemini, ensure the schema is Gemini-compatible
    if for_gemini:
        # Set a Gemini-compatible schema for the model
        PatchFunctionName.model_config = ConfigDict(
            json_schema_extra={
                "type": "OBJECT",
                "properties": {
                    "json_doc_id": {
                        "type": "STRING",
                        "description": "The ID of the function you are patching."
                    },
                    "reasoning": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "At least 2 logical reasons why this action ought to be taken."
                    },
                    "fixed_name": {
                        "type": "STRING",
                        "description": f"The corrected function name.{vname}"
                    }
                },
                "required": ["json_doc_id", "reasoning"]
            }
        )

    return PatchFunctionName


def _create_remove_doc_from_existing(existing: Union[dict, list]):
    if isinstance(existing, dict):
        existing_ids = set(existing)
    else:
        existing_ids = set()
        for schema_id, *_ in existing:
            existing_ids.add(schema_id)
    return _create_remove_doc_schema(tuple(sorted(existing_ids)))


@functools.lru_cache(maxsize=10)
def _create_remove_doc_schema(allowed_ids: tuple[str]) -> Type[BaseModel]:
    """Create a RemoveDoc schema that validates against a set of allowed IDs."""

    class RemoveDoc(BaseModel):
        """Use this tool to remove (delete) a doc by its ID."""

        json_doc_id: str = Field(
            ...,
            description=f"ID of the document to remove. Must be one of: {allowed_ids}",
        )

        @field_validator("json_doc_id")
        @classmethod
        def validate_doc_id(cls, v: str) -> str:
            if v not in allowed_ids:
                raise ValueError(
                    f"Document ID '{v}' not found. Available IDs: {sorted(allowed_ids)}"
                )
            return v

    RemoveDoc.__name__ = "RemoveDoc"
    return RemoveDoc

def _ensure_patches(args: dict) -> list[Dict[str, Any]]:
    """Process patches from different formats and ensure they're valid JsonPatch objects."""
    patches = args.get("patches", [])
    
    # If already a list, process it
    if isinstance(patches, list):
        processed_patches = []
        
        for patch in patches:
            if isinstance(patch, (dict, BaseModel)):
                # Extract required fields
                if isinstance(patch, BaseModel):
                    patch = patch.model_dump() if hasattr(patch, 'model_dump') else patch.dict()
                
                op = patch.get("op")
                path = patch.get("path")
                value = patch.get("value")
                
                # Verify required fields
                if op and path:
                    # For remove operations, value can be None
                    if op == "remove":
                        processed_patches.append({"op": op, "path": path})
                    # For add/replace operations, value is required
                    elif value is not None:
                        # Try to parse string values as JSON for complex values
                        parsed_value = value
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            try:
                                parsed_value = json.loads(value)
                            except json.JSONDecodeError:
                                # If parsing fails, use value as is
                                parsed_value = value
                                
                        processed_patches.append({
                            "op": op, 
                            "path": path, 
                            "value": parsed_value
                        })
        
        return processed_patches
    
    # Handle string format
    if isinstance(patches, str):
        try:
            # Direct JSON parsing attempt
            parsed = json.loads(patches)
            if isinstance(parsed, list):
                return _ensure_patches({"patches": parsed})
        except json.JSONDecodeError:
            # Fallback: Try to find a complete JSON array within the string
            bracket_depth = 0
            first_list_str = None
            start = patches.find("[")
            if start != -1:
                for i in range(start, len(patches)):
                    if patches[i] == "[":
                        bracket_depth += 1
                    elif patches[i] == "]":
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            first_list_str = patches[start : i + 1]
                            break
                if first_list_str:
                    try:
                        parsed = json.loads(first_list_str)
                        if isinstance(parsed, list):
                            return _ensure_patches({"patches": parsed})
                    except json.JSONDecodeError:
                        pass
    
    return []