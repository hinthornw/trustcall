# ruff: noqa: E501
from collections import defaultdict
from typing import Optional, Sequence

import pytest
from dydantic import create_model_from_schema
from langchain.chat_models import init_chat_model
from langsmith import aevaluate, expect, traceable
from langsmith.evaluation import EvaluationResults
from langsmith.schemas import Example, Run
from typing_extensions import TypedDict

from trustcall import ExtractionInputs, ExtractionOutputs, create_extractor


class Inputs(TypedDict, total=False):
    system_prompt: str
    input_str: str
    current_value: dict
    error_handling: list


class ContainsStr:
    def __init__(self, substr):
        self.substr = substr

    def __eq__(self, other):
        if not isinstance(other, str):
            return False
        return self.substr in other

    @classmethod
    def from_str(cls, s: str):
        return cls(s.split("ContainsStr:")[1])


class AnyStr(str):
    def __init__(self, matches: Sequence[str]):
        self.matches = matches

    def __hash__(self):
        return hash(tuple(self.matches))

    @classmethod
    def from_str(cls, s: str):
        return cls(s.split("AnyStr:")[1])


# Wrapper for my model
@traceable
async def predict_with_model(
    model_name: str, inputs: Inputs, tool_def: dict
) -> ExtractionOutputs:
    messages = [
        (
            "system",
            "Extract the relevant user preferences from the conversation."
            + inputs.get("system_prompt", ""),
        ),
        ("user", inputs["input_str"]),
    ]
    llm = init_chat_model(model_name, temperature=0.8)
    extractor = create_extractor(llm, tools=[tool_def], tool_choice=tool_def["name"])
    existing = inputs.get("current_value", {})
    extractor_inputs: dict = {"messages": messages}
    if existing:
        extractor_inputs["existing"] = {tool_def["name"]: existing}
    result = await extractor.ainvoke(ExtractionInputs(**extractor_inputs))
    # If you want, you can add scores inline
    expect.score(result["attempts"], key="num_attempts")
    return result


def score_run(run: Run, example: Example):
    results = []
    passed = True
    try:
        predicted = run.outputs["messages"][0].tool_calls[0]["args"]
        results.append(
            {
                "key": "valid_output",
                "score": 1,
            }
        )
    except Exception as e:
        passed = False
        results.extend(
            [
                {
                    "key": "valid_output",
                    "score": 0,
                    "comment": repr(e),
                },
                {
                    "key": "pass",
                    "score": 0,
                    "comment": "Failed to get valid output.",
                },
            ]
        )
        return {"results": results}
    schema = create_model_from_schema(example.inputs["tool_def"]["parameters"])
    try:
        schema.model_validate(predicted)
        results.append(
            {
                "key": "valid_schema",
                "score": 1,
            }
        )
    except Exception as e:
        passed = False
        results.append(
            {
                "key": "valid_schema",
                "score": 0,
                "comment": repr(e),
            }
        )

    if expected := example.outputs.get("expected"):
        try:
            for key, value in expected.items():
                pred = predicted[key]
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and sub_value.startswith(
                            "ContainsStr:"
                        ):
                            sub_value = ContainsStr.from_str(sub_value)
                        if sub_key.startswith("AnyStr:"):
                            sub_key = AnyStr.from_str(sub_key)
                            if not any(
                                pred.get(opt) == sub_value for opt in sub_key.matches
                            ):
                                raise AssertionError(
                                    f"Expected {sub_key} in {pred} to equal {sub_value}"
                                )
                        else:
                            assert pred.get(sub_key) == sub_value
                else:
                    assert pred == value
        except Exception as e:
            passed = False
            results.append(
                {
                    "key": "correct_output",
                    "score": 0,
                    "comment": repr(e),
                }
            )
    results.append(
        {
            "key": "pass",
            "score": passed,
        }
    )
    return {"results": results}


class DatasetInputs(TypedDict):
    inputs: Inputs
    tool_def: dict


class MetricProcessor:
    def __init__(self):
        self.counts = defaultdict(int)
        self.scores = defaultdict(float)

    def update(self, key: str, score: float):
        self.counts[key] += 1
        self.scores[key] += score

    def mean(self, key: str) -> Optional[float]:
        if key not in self.counts:
            return None
        return self.scores[key] / self.counts[key]

    def __getitem__(self, key: str):
        return self.mean(key)

    def __iter__(self):
        return {k: self[k] for k in self.counts.keys()}


@pytest.mark.asyncio_cooperative
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        # "gpt-3.5-turbo",
        # "claude-3-5-sonnet-20240620",
        # "accounts/fireworks/models/firefunction-v2",
    ],
)
async def test_model(model_name: str):
    if model_name == "accounts/fireworks/models/firefunction-v2":
        pytest.skip("this endpoint is too flakey")

    async def predict(dataset_inputs: DatasetInputs):
        return await predict_with_model(model_name, **dataset_inputs)

    result = await aevaluate(
        predict,
        data="trustcall",
        evaluators=[score_run],
        metadata={"model": model_name},
        experiment_prefix=f"{model_name}",
        max_concurrency=0,
    )
    processor = MetricProcessor()
    async for res in result:
        eval_results: EvaluationResults = res["evaluation_results"]
        for er in eval_results["results"]:
            processor.update(er.key, er.score)
    assert processor["pass"] > 0.99
