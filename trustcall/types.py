"""Type definitions for the trustcall package."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
)
from langchain_core.prompt_values import PromptValue
from typing_extensions import TypedDict


class SchemaInstance(tuple):
    """Represents an instance of a schema with its associated metadata.

    This named tuple is used to store information about a specific schema instance,
    including its unique identifier, the name of the schema it conforms to,
    and the actual data of the record.

    Attributes:
        record_id (str): A unique identifier for this schema instance.
        schema_name (str): The name of the schema that this instance conforms to.
        record (dict[str, Any]): The actual data of the record, stored as a dictionary.
    """

    record_id: str
    schema_name: str | Literal["__any__"]
    record: Dict[str, Any]

    def __new__(cls, record_id, schema_name, record):
        return tuple.__new__(cls, (record_id, schema_name, record))

    @property
    def record_id(self) -> str:
        return self[0]

    @property
    def schema_name(self) -> str | Literal["__any__"]:
        return self[1]

    @property
    def record(self) -> Dict[str, Any]:
        return self[2]


ExistingType = Union[
    Dict[str, Any], List[SchemaInstance], List[tuple[str, str, dict[str, Any]]]
]
"""Type for existing schemas.

Can be one of:
- Dict[str, Any]: A dictionary mapping schema names to schema instances.
- List[SchemaInstance]: A list of SchemaInstance named tuples.
- List[tuple[str, str, dict[str, Any]]]: A list of tuples containing
  (record_id, schema_name, record_dict).

This type allows for flexibility in representing existing schemas,
supporting both single and multiple instances of each schema type.
"""


class ExtractionInputs(TypedDict, total=False):
    messages: Union[Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]], PromptValue]
    existing: Optional[ExistingType]
    """Existing schemas. Key is the schema name, value is the schema instance.
    If a list, supports duplicate schemas to update.
    """


InputsLike = Union[ExtractionInputs, List[AnyMessage], PromptValue, str]


class ExtractionOutputs(TypedDict):
    messages: List[Any]  # AIMessage
    responses: List[Any]  # BaseModel
    response_metadata: List[dict[str, Any]]
    attempts: int


Message = Union[AnyMessage, MessageLikeRepresentation]

Messages = Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]]