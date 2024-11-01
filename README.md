# 🤝trustcall

![](_static/cover.png)

LLMs struggle when asked to generate or modify large JSON blobs. `trustcall` solves this by asking the LLM to generate [JSON patch](https://datatracker.ietf.org/doc/html/rfc6902) operations. This is a simpler task that can be done iteratively. This enables:

- ⚡ Faster & cheaper generation of structured output.
- 🐺Resilient retrying of validation errors, even for complex, nested schemas (defined as pydantic, schema dictionaries, or regular python functions)
- 🧩Acccurate updates to existing schemas, avoiding undesired deletions.

Works flexibly across a number of common LLM workflows like:

- ✂️ Extraction 
- 🧭 LLM routing
- 🤖 Multi-step agent tool use

## Installation

`pip install trustcall`

## Usage

- [Extracting complex schemas](#complex-schema)
- [Updating schemas](#updating-schemas)
- [Simultanous updates & insertions](#simultanous-updates--insertions)

## Why trustcall?

[Tool calling](https://python.langchain.com/docs/how_to/tool_calling/) makes it easier to compose LLM calls within reliable software systems, but LLM's today can be error prone and inefficient in two common scenarios:

1. Populating complex, nested schemas
2. Updating existing schemas without information loss

These problems are both exaggerated when you want to handle multiple tool calls.

Trustcall increases structured extraction reliability without restricting you to a subset of the JSON schema.

Let's see a couple examples to see what we mean.

### Complex schema

Take the following example:
<details>
    <summary>Schema definition</summary>

    from typing import List, Optional

    from pydantic import BaseModel


    class OutputFormat(BaseModel):
        preference: str
        sentence_preference_revealed: str


    class TelegramPreferences(BaseModel):
        preferred_encoding: Optional[List[OutputFormat]] = None
        favorite_telegram_operators: Optional[List[OutputFormat]] = None
        preferred_telegram_paper: Optional[List[OutputFormat]] = None


    class MorseCode(BaseModel):
        preferred_key_type: Optional[List[OutputFormat]] = None
        favorite_morse_abbreviations: Optional[List[OutputFormat]] = None


    class Semaphore(BaseModel):
        preferred_flag_color: Optional[List[OutputFormat]] = None
        semaphore_skill_level: Optional[List[OutputFormat]] = None


    class TrustFallPreferences(BaseModel):
        preferred_fall_height: Optional[List[OutputFormat]] = None
        trust_level: Optional[List[OutputFormat]] = None
        preferred_catching_technique: Optional[List[OutputFormat]] = None


    class CommunicationPreferences(BaseModel):
        telegram: TelegramPreferences
        morse_code: MorseCode
        semaphore: Semaphore


    class UserPreferences(BaseModel):
        communication_preferences: CommunicationPreferences
        trust_fall_preferences: TrustFallPreferences


    class TelegramAndTrustFallPreferences(BaseModel):
        pertinent_user_preferences: UserPreferences
</details>
    If you naively extract these values using `gpt-4o`, it's prone to failure:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
bound = llm.with_structured_output(TelegramAndTrustFallPreferences)

conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

bound.invoke(f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>""")
```

```
ValidationError: 1 validation error for TelegramAndTrustFallPreferences
pertinent_user_preferences.communication_preferences.semaphore
  Input should be a valid dictionary or instance of Semaphore [type=model_type, input_value=None, input_type=NoneType]
    For further information visit https://errors.pydantic.dev/2.8/v/model_type
```

If you try to use **strict** mode or OpenAI's `json_schema`, it will give you an error as well, since their parser doesn't support the complex JSON schemas:

```python
bound = llm.bind_tools([TelegramAndTrustFallPreferences], strict=True, response_format=TelegramAndTrustFallPreferences)

bound.invoke(f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>""")
```

```text
BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for function 'TelegramAndTrustFallPreferences': "}}
```

With `trustcall`, this extraction task is easy.

```python
from trustcall import create_extractor

bound = create_extractor(
    llm,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences",
)

result = bound.invoke(
    f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>"""
)
result["responses"][0]
```

```python
{
    "pertinent_user_preferences": {
        "communication_preferences": {
            "telegram": {
                "preferred_encoding": [
                    {
                        "preference": "morse",
                        "sentence_preference_revealed": "Morse, please.",
                    }
                ],
                "favorite_telegram_operators": None,
                "preferred_telegram_paper": [
                    {
                        "preference": "Daredevil",
                        "sentence_preference_revealed": 'Shall I use our "Daredevil" paper for this daring message?',
                    }
                ],
            },
            "morse_code": {
                "preferred_key_type": [
                    {
                        "preference": "straight key",
                        "sentence_preference_revealed": "I love using a straight key.",
                    }
                ],
                "favorite_morse_abbreviations": None,
            },
            "semaphore": {"preferred_flag_color": None, "semaphore_skill_level": None},
        },
        "trust_fall_preferences": {
            "preferred_fall_height": [
                {
                    "preference": "higher",
                    "sentence_preference_revealed": "I'm ready for a higher fall.",
                }
            ],
            "trust_level": None,
            "preferred_catching_technique": [
                {
                    "preference": "diamond formation",
                    "sentence_preference_revealed": "I prefer the diamond formation for catching.",
                }
            ],
        },
    }
}
```

What's different? `trustcall` handles prompt retries with a twist: rather than naively re-generating the full output, it prompts the LLM to generate a concise patch to fix the error in question. This is both **more reliable** than naive reprompting and **cheaper** since you only regenerate a subset of the full schema.

The "patch-don't-post" mantra affords us better performance in other ways too! Let's see how it helps **updates**.

### Updating schemas

Many tasks expect an LLM to correct or modify an existing object based on new information.

Take memory management as an example. Suppose you structure memories as JSON objects. When new information is provided, the LLM must reconcile this information with the existing document. Let's try this using naive regeneration of the document. We'll model memory as a single user profile:

```python
from typing import Dict, List, Optional

from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class Pet(BaseModel):
    kind: str
    name: Optional[str]
    age: Optional[int]


class Hobby(BaseModel):
    name: str
    skill_level: str
    frequency: str


class FavoriteMedia(BaseModel):
    shows: List[str]
    movies: List[str]
    books: List[str]


class User(BaseModel):
    preferred_name: str
    favorite_media: FavoriteMedia
    favorite_foods: List[str]
    hobbies: List[Hobby]
    age: int
    occupation: str
    address: Address
    favorite_color: Optional[str] = None
    pets: Optional[List[Pet]] = None
    languages: Dict[str, str] = {}
```

And set a starting profile state:
<details>
<summary>Starting profile</summary>

    initial_user = User(
        preferred_name="Alex",
        favorite_media=FavoriteMedia(
            shows=[
                "Friends",
                "Game of Thrones",
                "Breaking Bad",
                "The Office",
                "Stranger Things",
            ],
            movies=["The Shawshank Redemption", "Inception", "The Dark Knight"],
            books=["1984", "To Kill a Mockingbird", "The Great Gatsby"],
        ),
        favorite_foods=["sushi", "pizza", "tacos", "ice cream", "pasta", "curry"],
        hobbies=[
            Hobby(name="reading", skill_level="expert", frequency="daily"),
            Hobby(name="hiking", skill_level="intermediate", frequency="weekly"),
            Hobby(name="photography", skill_level="beginner", frequency="monthly"),
            Hobby(name="biking", skill_level="intermediate", frequency="weekly"),
            Hobby(name="swimming", skill_level="expert", frequency="weekly"),
            Hobby(name="canoeing", skill_level="beginner", frequency="monthly"),
            Hobby(name="sailing", skill_level="intermediate", frequency="monthly"),
            Hobby(name="weaving", skill_level="beginner", frequency="weekly"),
            Hobby(name="painting", skill_level="intermediate", frequency="weekly"),
            Hobby(name="cooking", skill_level="expert", frequency="daily"),
        ],
        age=28,
        occupation="Software Engineer",
        address=Address(
            street="123 Tech Lane", city="San Francisco", country="USA", postal_code="94105"
        ),
        favorite_color="blue",
        pets=[Pet(kind="cat", name="Luna", age=3)],
        languages={"English": "native", "Spanish": "intermediate", "Python": "expert"},
    )
</details>

Giving the following conversation, we'd expect the memory to be **expanded** to include video gaming but not drop any other information:

```python

conversation = """Friend: Hey Alex, how's the new job going? I heard you switched careers recently.
Alex: It's going great! I'm loving my new role as a Data Scientist. The work is challenging but exciting. I've moved to a new apartment in New York to be closer to the office.
Friend: That's a big change! Are you still finding time for your hobbies?
Alex: Well, I've had to cut back on some. I'm not doing much sailing or canoeing these days. But I've gotten really into machine learning projects in my free time. I'd say I'm getting pretty good at it - probably an intermediate level now.
Friend: Sounds like you're keeping busy! How's Luna doing?
Alex: Oh, Luna's great. She just turned 4 last week. She's actually made friends with my new pet, Max the dog. He's a playful 2-year-old golden retriever.
Friend: Two pets now! That's exciting. Hey, want to catch the new season of Stranger Things this weekend?
Alex: Actually, I've kind of lost interest in that show. But I'm really into this new series called "The Mandalorian". We could watch that instead! Oh, and I recently watched "Parasite" - it's become one of my favorite movies.
Friend: Sure, that sounds fun. Should I bring some food? I remember you love sushi.
Alex: Sushi would be perfect! Or maybe some Thai food - I've been really into that lately. By the way, I've been practicing my French. I'd say I'm at a beginner level now.
Friend: That's great! You're always learning something new. How's the cooking going?
Alex: It's going well! I've been cooking almost every day now. I'd say I've become quite proficient at it."""


# Naive approach
bound = llm.with_structured_output(User)
naive_result = bound.invoke(
    f"""Update the memory (JSON doc) to incorporate new information from the following conversation:
<user_info>
{initial_user.model_dump()}
</user_info>
<convo>
{conversation}
</convo>"""
)
print("Naive approach result:")
naive_output = naive_result.model_dump()
print(naive_output)
```

<details>
    <summary>Naive output</summary>
    {
        "preferred_name": "Alex",
        "favorite_media": {
            "shows": ["Friends", "Game of Thrones", "Breaking Bad", "The Office"],
            "movies": [
                "The Shawshank Redemption",
                "Inception",
                "The Dark Knight",
                "Parasite",
            ],
            "books": ["1984", "To Kill a Mockingbird", "The Great Gatsby"],
        },
        "favorite_foods": [
            "sushi",
            "pizza",
            "tacos",
            "ice cream",
            "pasta",
            "curry",
            "Thai food",
        ],
        "hobbies": [
            {"name": "reading", "skill_level": "expert", "frequency": "daily"},
            {"name": "hiking", "skill_level": "intermediate", "frequency": "weekly"},
            {"name": "photography", "skill_level": "beginner", "frequency": "monthly"},
            {"name": "biking", "skill_level": "intermediate", "frequency": "weekly"},
            {"name": "swimming", "skill_level": "expert", "frequency": "weekly"},
            {"name": "weaving", "skill_level": "beginner", "frequency": "weekly"},
            {"name": "painting", "skill_level": "intermediate", "frequency": "weekly"},
            {"name": "cooking", "skill_level": "expert", "frequency": "daily"},
            {
                "name": "machine learning projects",
                "skill_level": "intermediate",
                "frequency": "free time",
            },
        ],
        "age": 28,
        "occupation": "Data Scientist",
        "address": {
            "street": "New Apartment",
            "city": "New York",
            "country": "USA",
            "postal_code": "unknown",
        },
        "favorite_color": "blue",
        "pets": [
            {"kind": "cat", "name": "Luna", "age": 4},
            {"kind": "dog", "name": "Max", "age": 2},
        ],
        "languages": {},
    }

</details>

You'll notice that all the "languages" section was dropped here, and "The Mandalorian" was omitted. Alex may be injured, but he didn't forget how to speak!

When you run this code, it's _possible_ it will get it right: LLMs are stochastic after all (which is a good thing). And you could definitely prompt engineer it to be more reliable, but **that's not good enough.**

For memory management, you will be updating objects **constantly**, and it's still **too easy** for LLMs to "accidentally" omit information when generating updates, or to miss content in the conversation.

`trustcall` lets the LLM **focus on what has changed**.

```python
# Trustcall approach
from trustcall import create_extractor

bound = create_extractor(llm, tools=[User])

trustcall_result = bound.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"""Update the memory (JSON doc) to incorporate new information from the following conversation:
<convo>
{conversation}
</convo>""",
            }
        ],
        "existing": {"User": initial_user.model_dump()},
    }
)
print("\nTrustcall approach result:")
trustcall_output = trustcall_result["responses"][0].model_dump()
print(trustcall_output)
```
Output:

<details>
    <summary>`trustcall` output</summary>

    {
    "preferred_name": "Alex",
    "favorite_media": {
        "shows": [
            "Friends",
            "Game of Thrones",
            "Breaking Bad",
            "The Office",
            "The Mandalorian",
        ],
        "movies": [
            "The Shawshank Redemption",
            "Inception",
            "The Dark Knight",
            "Parasite",
        ],
        "books": ["1984", "To Kill a Mockingbird", "The Great Gatsby"],
    },
    "favorite_foods": [
        "sushi",
        "pizza",
        "tacos",
        "ice cream",
        "pasta",
        "curry",
        "Thai food",
    ],
    "hobbies": [
        {"name": "reading", "skill_level": "expert", "frequency": "daily"},
        {"name": "hiking", "skill_level": "intermediate", "frequency": "weekly"},
        {"name": "photography", "skill_level": "beginner", "frequency": "monthly"},
        {"name": "biking", "skill_level": "intermediate", "frequency": "weekly"},
        {"name": "swimming", "skill_level": "expert", "frequency": "weekly"},
        {"name": "weaving", "skill_level": "beginner", "frequency": "weekly"},
        {"name": "painting", "skill_level": "intermediate", "frequency": "weekly"},
        {"name": "cooking", "skill_level": "expert", "frequency": "daily"},
        {
            "name": "machine learning projects",
            "skill_level": "intermediate",
            "frequency": "daily",
        },
    ],
    "age": 28,
    "occupation": "Data Scientist",
    "address": {
        "street": "New Apartment",
        "city": "New York",
        "country": "USA",
        "postal_code": "10001",
    },
    "favorite_color": "blue",
    "pets": [
        {"kind": "cat", "name": "Luna", "age": 4},
        {"kind": "dog", "name": "Max", "age": 2},
    ],
    "languages": {
        "English": "native",
        "Spanish": "intermediate",
        "Python": "expert",
        "French": "beginner",
    },
    }
</details>

No fields omitted, and the important new information is seamlessly integrated.

### Simultanous updates & insertions

Both problems above (difficulty with type-safe generation of complex schemas & difficulty with generating the correct edits to existing schemas) are compounded when you have to be prompting the LLM to handle **both** updates **and** inserts, as is often the case when extracting mulptiple memory "events" from conversations.

Let's see an example below. Suppose you are managing a list of "relationships":

```python
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    """Someone the user knows or interacts with."""

    name: str
    relationship: str = Field(description="How they relate to the user.")

    notes: List[str] = Field(
        description="Memories and other observations about the person"
    )


# Initial data
initial_people = [
    Person(
        name="Emma Thompson",
        relationship="College friend",
        notes=["Loves hiking", "Works in marketing", "Has a dog named Max"],
    ),
    Person(
        name="Michael Chen",
        relationship="Coworker",
        notes=["Great at problem-solving", "Vegetarian", "Plays guitar"],
    ),
    Person(
        name="Sarah Johnson",
        relationship="Neighbor",
        notes=["Has two kids", "Loves gardening", "Makes amazing cookies"],
    ),
]

# Convert to the format expected by the extractor
existing_data = [
    (str(i), "Person", person.model_dump()) for i, person in enumerate(initial_people)
]
```

```python
conversation = """
Me: I ran into Emma Thompson at the park yesterday. She was walking her new puppy, a golden retriever named Sunny. She mentioned she got promoted to Senior Marketing Manager last month.
Friend: That's great news for Emma! How's she enjoying the new role?
Me: She seems to be thriving. Oh, and did you know she's taken up rock climbing? She invited me to join her at the climbing gym sometime.
Friend: Wow, rock climbing? That's quite a change from hiking. Speaking of friends, have you heard from Michael Chen recently?
Me: Actually, yes. We had a video call last week. He's switched jobs and is now working as a Data Scientist at a startup. He's also mentioned he's thinking of going vegan.
Friend: That's a big change for Michael! Both career and diet-wise. How about your neighbor, Sarah? Is she still teaching?
Me: Sarah's doing well. Her kids are growing up fast - her oldest just started middle school. She's still teaching, but now she's focusing on special education. She's really passionate about it.
Friend: That's wonderful. Oh, before I forget, I wanted to introduce you to my cousin who just moved to town. Her name is Olivia Davis, she's a 27-year-old graphic designer. She's looking to meet new people and expand her social circle. I thought you two might get along well.
Me: That sounds great! I'd love to meet her. Maybe we could all get together for coffee next week?
Friend: Perfect! I'll set it up. Olivia loves art and is always sketching in her free time. She also volunteers at the local animal shelter on weekends.
"""

from langchain_openai import ChatOpenAI

# Now, let's use the extractor to update existing entries and create new ones
from trustcall import create_extractor

llm = ChatOpenAI(model="gpt-4o")

extractor = create_extractor(
    llm,
    tools=[Person],
    tool_choice="any",
    enable_inserts=True,
)

result = extractor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"Update existing person records and create new ones based on the following conversation:\n\n{conversation}",
            }
        ],
        "existing": existing_data,
    }
)

# Print the results
print("Updated and new person records:")
for r, rmeta in zip(result["responses"], result["response_metadata"]):
    print(f"ID: {rmeta.get('json_doc_id', 'New')}")
    print(r.model_dump_json(indent=2))
    print()
```
The LLM is able to update existing values while also inserting new ones!

```text
Updated and new person records:
ID: 0
{
  "name": "Emma Thompson",
  "relationship": "College friend",
  "notes": [
    "Loves hiking",
    "Works in marketing",
    "Has a dog named Max",
    "Walking her new puppy, a golden retriever named Sunny",
    "Promoted to Senior Marketing Manager",
    "Taken up rock climbing"
  ]
}

ID: 1
{
  "name": "Michael Chen",
  "relationship": "Coworker",
  "notes": [
    "Great at problem-solving",
    "Vegetarian",
    "Plays guitar",
    "Working as a Data Scientist at a startup",
    "Thinking of going vegan"
  ]
}

ID: 2
{
  "name": "Sarah Johnson",
  "relationship": "Neighbor",
  "notes": [
    "Has two kids",
    "Loves gardening",
    "Makes amazing cookies",
    "Oldest child started middle school",
    "Focusing on special education",
    "Passionate about teaching"
  ]
}

ID: New
{
  "name": "Olivia Davis",
  "relationship": "Friend's cousin",
  "notes": [
    "27-year-old graphic designer",
    "Looking to meet new people",
    "Loves art and sketching",
    "Volunteers at the local animal shelter on weekends"
  ]
}
```

## More Examples

Trustcall works out of the box with any tool-calling LLM from the LangChain ecosystem.

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
you can easily use the abstraction for conversational agent applications:

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

## Evaluating

We have a simple evaluation benchmark in [test_evals.py](./tests/evals/test_evals.py).

To run, first clone the dataset

```python
from langsmith import Client

Client().clone_public_dataset("https://smith.langchain.com/public/0544c02f-9617-4095-bc15-3a9af1189819/d")
```

Then run the evals:

```bash
make evals
```

This requires some additional dependencies, as well as API keys for the models being compared.
