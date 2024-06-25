# trustcall

Tool calling & validated extraction you can trust, built on LangGraph.

Uses [patch](https://datatracker.ietf.org/doc/html/rfc6902)-based extraction for:

- Faster & cheaper generation of structured output.
- Resilient retrying of validation errors, even for complex, nested schemas.
- Acccurate updates to existing schemas, avoiding undesired deletions.

Works flexibly across a number of common LLM workflows:

1. Extraction
2. LLM routing
3. Multi-step agent tool use

and more!

## Examples

First, install:

```bash
pip install -U trustcall langchain-fireworks
```

Then set up your schema:

```python
from typing import List

from langchain_fireworks import ChatFireworks
from pydantic.v1 import BaseModel, Field, validator
from trustcall import create_extractor


class Preferences(BaseModel):
    foods: List[str] = Field(description="Favorite foods")

    @validator("foods")
    def at_least_three_foods(cls, v):
        # Just a silly example to show how it can recover from a
        # validation error.
        if len(v) < 3:
            raise ValueError("Must have at least three favorite foods")
        return v


llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v2")

extractor = create_extractor(llm, tools=[Preferences], tool_choice="Preferences")
res = extractor.invoke({"messages": [("user", "I like apple pie and ice cream.")]})
msg = res["messages"][-1]
print(msg.tool_calls)
print(res["responses"])
# [{'id': 'call_pBrHTBNHNLnGCv7UBKBJz6xf', 'name': 'Preferences', 'args': {'foods': ['apple pie', 'ice cream', 'pizza', 'sushi']}}]
# [Preferences(foods=['apple pie', 'ice cream', 'pizza', 'sushi'])]
```

Since the extractor also returns the chat message (with validated and cleaned tools),
you can easiliy use the abstraction for conversational agent applications:

```python
import operator
from datetime import datetime
from typing import List

import pytz
from langchain_fireworks import ChatFireworks
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic.v1 import BaseModel, Field, validator
from trustcall import create_extractor
from typing_extensions import Annotated, TypedDict


class Preferences(BaseModel):
    foods: List[str] = Field(description="Favorite foods")

    @validator("foods")
    def at_least_three_foods(cls, v):
        if len(v) < 3:
            raise ValueError("Must have at least three favorite foods")
        return v


llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v2")


def save_user_information(preferences: Preferences):
    """Save user information to a database."""
    return "User information saved"


def lookup_time(tz: str) -> str:
    """Lookup the current time in a given timezone."""
    try:
        # Convert the timezone string to a timezone object
        timezone = pytz.timezone(tz)
        # Get the current time in the given timezone
        tm = datetime.now(timezone)
        return f"The current time in {tz} is {tm.strftime('%H:%M:%S')}"
    except pytz.UnknownTimeZoneError:
        return f"Unknown timezone: {tz}"


agent = create_extractor(llm, tools=[save_user_information, lookup_time])


class State(TypedDict):
    messages: Annotated[list, operator.add]


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([save_user_information, lookup_time]))
builder.add_edge("tools", "agent")
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1234"}}
res = graph.invoke({"messages": [("user", "Hi there!")]}, config)
res["messages"][-1].pretty_print()
# ================================== Ai Message ==================================

# I'm happy to help you with any questions or tasks you have. What's on your mind today?
res = graph.invoke(
    {"messages": [("user", "Curious; what's the time in denver right now?")]}, config
)
res["messages"][-1].pretty_print()
# ================================== Ai Message ==================================

# The current time in Denver is 00:57:25.
res = graph.invoke(
    {
        "messages": [
            ("user", "Did you know my favorite foods are spinach and potatoes?")
        ]
    },
    config,
)
res["messages"][-1].pretty_print()
# ================================== Ai Message ==================================

# I've saved your favorite foods, spinach and potatoes.

```

If you check out the [last call in that conversation](https://smith.langchain.com/public/b83d6db1-ffb9-4817-a166-bbc5004bbc25/r/5a05f73b-1d7e-47d4-9e40-0e8aaa3faa28), you can see that the agent initially generated an invalid tool call, but our validation was able to fix up the output before passing the payload on to our tools.

These are just a couple examples to highlight what you can accomplish with `trustcall`.

#### Explanation

You can write this yourself (I wrote and tested this in a few hours, but I bet you're faster)!

To reproduce the basic logic of the library, simply:

1. Prompt the LLM to generate parameters for the schemas of zero or more tools.
2. If any of these schemas raise validation errors, re-prompt the LLM to fix by generating a JSON Patch.

The extractor also accepts a dictionary of **existing** schemas it can update (for situations where you have some structured
representation of an object and you want to extend or update parts of it using new information.)

The dictionary format is `**schema_name**: **current_schema**`.

In this case, the logic is simpler:

1. Prompt the LLM to generate one or more JSON Patches for any (or all) of the existing schemas.
2. After applying the patches, if any of these schemas are invalid, re-prompt the LLM to fix using more patches.

`trustcall` also uses + extends some convenient utilities to let you define schemas in several ways:

1. Regular python functions (with typed arguments to apply the validation).
2. Pydantic objects
3. JSON schemas (we will still validate your calls using the schemas' typing and constraints).

as well as providing support for `langchain-core`'s tools.
