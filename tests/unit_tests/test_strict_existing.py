import logging

import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from trustcall._base import (
    SchemaInstance,
    _ExtractUpdates,
)

logger = logging.getLogger(__name__)


class DummySchema(BaseModel):
    """A dummy recognized tool schema for testing."""

    foo: str
    bar: int


@pytest.mark.parametrize(
    "existing_schema_policy,existing,expect_error,expected_coerced",
    [
        #
        # Case 1: Single recognized key => always OK in all modes
        #
        (
            True,  # strict mode
            {"DummySchema": {"foo": "val", "bar": 123}},
            False,  # no error
            {"DummySchema": {"foo": "val", "bar": 123}},
        ),
        (
            False,  # non-strict mode
            {"DummySchema": {"foo": "val", "bar": 123}},
            False,
            {"DummySchema": {"foo": "val", "bar": 123}},
        ),
        (
            "ignore",  # ignore mode
            {"DummySchema": {"foo": "val", "bar": 123}},
            False,
            {"DummySchema": {"foo": "val", "bar": 123}},
        ),
        #
        # Case 2: Single unknown key
        #
        (
            True,  # strict => error
            {"UnknownSchema": {"foo": "val", "bar": 999}},
            True,
            None,
        ),
        (
            False,  # non-strict => fallback to __any__
            {"UnknownSchema": {"foo": "val", "bar": 999}},
            False,
            {"UnknownSchema": {"foo": "val", "bar": 999}},
        ),
        (
            "ignore",  # ignore => skip
            {"UnknownSchema": {"foo": "val", "bar": 999}},
            False,
            {},
        ),
        #
        # Case 3: Mixed known + unknown schemas
        #
        (
            True,  # strict => error on unknown
            {
                "DummySchema": {"foo": "val", "bar": 321},
                "RandomThing": {"some": "stuff"},
                "__any__": {"whatever": "stuff"},
            },
            True,
            None,
        ),
        (
            False,  # non-strict => fallback unknown to __any__
            {
                "DummySchema": {"foo": "val2", "bar": 789},
                "RandomThing": {"some": "stuff2"},
                "__any__": {"pass_through": True},
            },
            False,
            {
                "DummySchema": {"foo": "val2", "bar": 789},
                "RandomThing": {"some": "stuff2"},
                "__any__": {"pass_through": True},
            },
        ),
        (
            "ignore",  # ignore => skip unknown
            {
                "DummySchema": {"foo": "val3", "bar": 456},
                "RandomThing": {"some": "stuff3"},
            },
            False,
            {
                "DummySchema": {"foo": "val3", "bar": 456},
            },
        ),
        #
        # Case 4: List-based format - single recognized schema
        #
        (
            True,  # strict
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
            False,
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
        ),
        (
            False,  # non-strict
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
            False,
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
        ),
        (
            "ignore",  # ignore
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
            False,
            [SchemaInstance("rid", "DummySchema", {"foo": "abc", "bar": 42})],
        ),
        #
        # Case 5: List-based - single unknown schema
        #
        (
            True,  # strict => error
            [SchemaInstance("rid2", "Unknown", {"some": "stuff"})],
            True,
            None,
        ),
        (
            False,  # non-strict => fallback to __any__
            [SchemaInstance("rid2", "Unknown", {"some": "stuff"})],
            False,
            [SchemaInstance("rid2", "Unknown", {"some": "stuff"})],
        ),
        (
            "ignore",  # ignore => skip
            [SchemaInstance("rid2", "Unknown", {"some": "stuff"})],
            False,
            [],
        ),
        #
        # Case 6: List-based - explicit __any__ schema
        #
        (
            True,  # strict - __any__ is always allowed
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
            False,
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
        ),
        (
            False,  # non-strict
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
            False,
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
        ),
        (
            "ignore",  # ignore
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
            False,
            [SchemaInstance("rid3", "__any__", {"some": "stuff"})],
        ),
        #
        # Case 7: List-based - mixed schemas
        #
        (
            True,  # strict => error on unknown
            [
                SchemaInstance("rid4", "DummySchema", {"foo": "test", "bar": 100}),
                SchemaInstance("rid5", "Unknown", {"data": "test"}),
                SchemaInstance("rid6", "__any__", {"other": "data"}),
            ],
            True,
            None,
        ),
        (
            False,  # non-strict => fallback unknown to __any__
            [
                SchemaInstance("rid4", "DummySchema", {"foo": "test", "bar": 100}),
                SchemaInstance("rid5", "Unknown", {"data": "test"}),
                SchemaInstance("rid6", "__any__", {"other": "data"}),
            ],
            False,
            [
                SchemaInstance("rid4", "DummySchema", {"foo": "test", "bar": 100}),
                SchemaInstance("rid5", "Unknown", {"data": "test"}),
                SchemaInstance("rid6", "__any__", {"other": "data"}),
            ],
        ),
        (
            "ignore",  # ignore => skip unknown
            [
                SchemaInstance("rid4", "DummySchema", {"foo": "test", "bar": 100}),
                SchemaInstance("rid5", "Unknown", {"data": "test"}),
                SchemaInstance("rid6", "__any__", {"other": "data"}),
            ],
            False,
            [
                SchemaInstance("rid4", "DummySchema", {"foo": "test", "bar": 100}),
                SchemaInstance("rid6", "__any__", {"other": "data"}),
            ],
        ),
        #
        # Case 8: Tuple format
        #
        (
            True,  # strict
            [("rid7", "DummySchema", {"foo": "tuple", "bar": 200})],
            False,
            [SchemaInstance("rid7", "DummySchema", {"foo": "tuple", "bar": 200})],
        ),
        (
            True,  # strict => error on unknown
            [("rid8", "Unknown", {"data": "test"})],
            True,
            None,
        ),
        (
            False,  # non-strict => fallback
            [("rid8", "Unknown", {"data": "test"})],
            False,
            [SchemaInstance("rid8", "Unknown", {"data": "test"})],
        ),
        (
            "ignore",  # ignore => skip
            [("rid8", "Unknown", {"data": "test"})],
            False,
            [],
        ),
        #
        # Case 9: Invalid type - always error regardless of mode
        #
        (
            True,
            "invalid_type",
            True,
            None,
        ),
        (
            False,
            "invalid_type",
            True,
            None,
        ),
        (
            "ignore",
            "invalid_type",
            True,
            None,
        ),
    ],
)
def test_validate_existing_strictness(
    existing_schema_policy, existing, expect_error, expected_coerced
):
    """Test various scenarios of validation."""
    tools = {"DummySchema": DummySchema}
    llm = ChatOpenAI(model="gpt-4o-mini")
    extractor = _ExtractUpdates(
        llm=llm,  # We won't actually call the LLM here but we need it for parsing.
        tools=tools,
        enable_inserts=False,
        existing_schema_policy=existing_schema_policy,
    )

    if expect_error:
        with pytest.raises(ValueError) as exc:
            extractor._validate_existing(existing)
        # Check error message contains expected text
        error_msg = str(exc.value).lower()
        assert any(
            text in error_msg
            for text in [
                "doesn't match any",
                "does not match any",
                "invalid type",
                "unknown schema",
            ]
        ), f"Unexpected error message: {error_msg}"
        return

    coerced = extractor._validate_existing(existing)

    # Handle different return types based on input type
    if isinstance(existing, dict):
        assert isinstance(coerced, dict)
        if expected_coerced is not None:
            assert coerced == expected_coerced

    elif isinstance(existing, list):
        assert isinstance(coerced, list)
        if expected_coerced is not None:
            assert len(coerced) == len(expected_coerced)
            for c, e in zip(coerced, expected_coerced):
                assert isinstance(c, SchemaInstance)
                assert c.record_id == e.record_id
                assert c.schema_name == e.schema_name
                assert c.record == e.record

    # Additional validation based on existing_schema_policy mode
    if existing_schema_policy == "ignore":
        # Verify that unknown schemas are not present
        if isinstance(coerced, dict):
            assert all(k in tools or k == "__any__" for k in coerced)
        elif isinstance(coerced, list):
            assert all(s.schema_name in tools or s.schema_name == "__any__" for s in coerced)
    elif existing_schema_policy is False:
        pass 
