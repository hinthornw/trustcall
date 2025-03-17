"""Facade module for the trustcall package.

This module re-exports the key functionality from the other modules in the package.
It serves as the main entry point to the library, providing a simplified interface
for users.
"""

# Re-export key functionality
from trustcall.extract import (
    create_extractor,
)
from trustcall.states import (
    ExtractionState,
    ExtendedExtractState,
    DeletionState,
)
from trustcall.tools import ensure_tools, _convert_any_typed_dicts_to_pydantic
from trustcall.types import ExtractionInputs, ExtractionOutputs, SchemaInstance
from trustcall.schema import _create_patch_doc_schema, _create_patch_function_errors_schema
from trustcall.extract import _Extract, _ExtractUpdates
from trustcall.patch import _Patch

# Create default versions of PatchDoc and PatchFunctionErrors for backward compatibility
PatchDoc = _create_patch_doc_schema(for_gemini=False)
PatchFunctionErrors = _create_patch_function_errors_schema(for_gemini=False)

__all__ = [
    "create_extractor",
    "ensure_tools",
    "ExtractionInputs",
    "ExtractionOutputs",
    "ExtractionState",
    "ExtendedExtractState",
    "DeletionState",
    "SchemaInstance",
    "PatchDoc",
    "PatchFunctionErrors",
    "_ExtractUpdates",
    "_Extract",
    "_Patch",
    "_convert_any_typed_dicts_to_pydantic",
]