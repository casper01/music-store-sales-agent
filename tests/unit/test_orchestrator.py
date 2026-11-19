import json

import pytest

from sales_agent.agent.contracts import StopReason, ToolCallRequest
from sales_agent.agent.orchestrator import (
    EMPTY_RESPONSE_ANSWER,
    ITERATION_LIMIT_ANSWER,
    AgentOrchestrator,
)
from sales_agent.errors import ModelError, ToolExecutionError, ToolInputError

from fakes import ScriptedModel, answer, calls, tool_call


class RecordingTool:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.description = f"{name} test tool"
        self.parameters = {"type": "object", "properties": {}, "additionalProperties": True}
        self.result = result if result is not None else {"value": name}
        self.error = error
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result


def tool_messages(model, request_index):
    messages = model.requests[request_index][0]
    return [message for message in messages if message["role"] == "tool"]


def test_answers_directly_when_model_chooses_no_tool():
    sql = RecordingTool("run_sql")
    model = ScriptedModel([answer("  I cannot change data.  ")])

    result = AgentOrchestrator(model, [sql], max_iterations=3).run("Delete everything")

    assert result.answer == "I cannot change data."
    assert result.stop_reason is StopReason.ANSWERED
    assert result.tool_calls == ()
    assert sql.calls == []


def test_dispatches_the_tool_chosen_by_the_model_and_returns_its_output():
    sql = RecordingTool("run_sql", result={"rows": [{"n": 1}]})
    knowledge = RecordingTool("search_knowledge")
    model = ScriptedModel([calls(tool_call("run_sql", {"query": "SELECT 1"})), answer("One row.")])

    result = AgentOrchestrator(model, [sql, knowledge], max_iterations=3).run("Count")

    assert result.answer == "One row."
    assert result.tool_names == ["run_sql"]
    assert sql.calls == [{"query": "SELECT 1"}]
    assert knowledge.calls == []
    [message] = tool_messages(model, 1)
    assert message["tool_call_id"] == "call-1"
    assert json.loads(message["content"]) == {"rows": [{"n": 1}]}


def test_sends_system_prompt_question_and_tool_definitions():
    model = ScriptedModel([answer("ok")])
    AgentOrchestrator(model, [RecordingTool("run_sql")], max_iterations=1).run("  Question?  ")

    messages, tools = model.requests[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Question?"}
    assert tools[0]["function"]["name"] == "run_sql"
    assert tools[0]["function"]["parameters"]["type"] == "object"


def test_handles_parallel_calls_and_multiple_iterations():
    sql = RecordingTool("run_sql")
    knowledge = RecordingTool("search_knowledge")
    model = ScriptedModel(
        [
            calls(
                tool_call("get_schema", {}, "c1"),
                tool_call("search_knowledge", {"query": "revenue"}, "c2"),
            ),
            calls(tool_call("run_sql", {"query": "SELECT 1"}, "c3")),
            answer("Combined answer."),
        ]
    )
    schema = RecordingTool("get_schema")

    result = AgentOrchestrator(model, [schema, sql, knowledge], max_iterations=5).run("Q")

    assert result.answer == "Combined answer."
    assert result.iterations == 3
    assert result.tool_names == ["get_schema", "search_knowledge", "run_sql"]
    assert [m["tool_call_id"] for m in tool_messages(model, 2)] == ["c1", "c2", "c3"]
    assistant_turn = model.requests[1][0][2]
    assert [call["id"] for call in assistant_turn["tool_calls"]] == ["c1", "c2"]


def test_stops_at_iteration_limit_without_fabricating_an_answer():
    tool = RecordingTool("run_sql")
    looping = calls(tool_call("run_sql", {"query": "SELECT 1"}))
    model = ScriptedModel([looping, looping, looping])

    result = AgentOrchestrator(model, [tool], max_iterations=2).run("Q")

    assert result.answer == ITERATION_LIMIT_ANSWER
    assert result.stop_reason is StopReason.ITERATION_LIMIT
    assert len(model.requests) == 2
    assert len(tool.calls) == 2


@pytest.mark.parametrize(
    ("request_", "expected_error"),
    [
        (tool_call("unknown_tool", {}), "Unknown tool"),
        (tool_call("run_sql", "{not json"), "not valid JSON"),
        (tool_call("run_sql", "[1, 2]"), "JSON object"),
    ],
)
def test_malformed_tool_calls_are_reported_back_to_the_model(request_, expected_error):
    tool = RecordingTool("run_sql")
    model = ScriptedModel([calls(request_), answer("Recovered.")])

    result = AgentOrchestrator(model, [tool], max_iterations=3).run("Q")

    assert result.answer == "Recovered."
    assert tool.calls == []
    assert result.tool_calls[0].ok is False
    assert expected_error in json.loads(tool_messages(model, 1)[0]["content"])["error"]


@pytest.mark.parametrize(
    "error", [ToolInputError("bad argument"), ToolExecutionError("query timed out")]
)
def test_tool_errors_are_returned_to_the_model_so_it_can_retry(error):
    tool = RecordingTool("run_sql", error=error)
    model = ScriptedModel([calls(tool_call("run_sql", {"query": "x"})), answer("Explained.")])

    result = AgentOrchestrator(model, [tool], max_iterations=3).run("Q")

    assert result.tool_calls[0].error == str(error)
    assert json.loads(tool_messages(model, 1)[0]["content"]) == {"error": str(error)}


def test_unexpected_tool_exceptions_are_not_swallowed():
    tool = RecordingTool("run_sql", error=KeyError("bug"))
    model = ScriptedModel([calls(tool_call("run_sql", {}))])

    with pytest.raises(KeyError):
        AgentOrchestrator(model, [tool], max_iterations=3).run("Q")


def test_model_errors_propagate_to_the_caller():
    class BrokenModel:
        def complete(self, messages, tools):
            raise ModelError("timeout")

    with pytest.raises(ModelError):
        AgentOrchestrator(BrokenModel(), [], max_iterations=1).run("Q")


def test_empty_model_response_is_reported_honestly():
    result = AgentOrchestrator(ScriptedModel([answer("   ")]), [], max_iterations=2).run("Q")
    assert result.answer == EMPTY_RESPONSE_ANSWER
    assert result.stop_reason is StopReason.EMPTY_RESPONSE


def test_oversized_tool_output_is_cut_with_a_note():
    tool = RecordingTool("run_sql", result={"blob": "x" * 5000})
    model = ScriptedModel([calls(tool_call("run_sql", {})), answer("ok")])

    AgentOrchestrator(model, [tool], max_iterations=2, max_tool_output_chars=1000).run("Q")

    payload = json.loads(tool_messages(model, 1)[0]["content"])
    assert len(payload["partial_output"]) == 1000
    assert "narrow" in payload["note"]


def test_observer_receives_every_tool_call():
    seen = []
    model = ScriptedModel([calls(tool_call("run_sql", {"query": "q"})), answer("ok")])
    AgentOrchestrator(
        model, [RecordingTool("run_sql")], max_iterations=2, on_tool_call=seen.append
    ).run("Q")
    assert [(record.name, record.ok) for record in seen] == [("run_sql", True)]


def test_model_call_observer_sees_the_growing_conversation():
    sizes = []
    model = ScriptedModel([calls(tool_call("run_sql", {"query": "q"})), answer("ok")])
    AgentOrchestrator(
        model,
        [RecordingTool("run_sql")],
        max_iterations=2,
        on_model_call=lambda iteration, messages: sizes.append((iteration, len(messages))),
    ).run("Q")
    # system + user, then + assistant tool call + tool result
    assert sizes == [(1, 2), (2, 4)]


@pytest.mark.parametrize("question", ["", "   "])
def test_rejects_empty_question(question):
    with pytest.raises(ValueError):
        AgentOrchestrator(ScriptedModel([]), [], max_iterations=1).run(question)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="positive"):
        AgentOrchestrator(ScriptedModel([]), [], max_iterations=0)
    with pytest.raises(ValueError, match="unique"):
        AgentOrchestrator(
            ScriptedModel([]), [RecordingTool("a"), RecordingTool("a")], max_iterations=1
        )


def test_provider_fields_of_tool_calls_are_sent_back_to_the_model():
    signed = ToolCallRequest(
        id="c1",
        name="run_sql",
        arguments_json="{}",
        provider_fields={"extra_content": {"google": {"thought_signature": "sig"}}},
    )
    model = ScriptedModel([calls(signed), answer("ok")])

    AgentOrchestrator(model, [RecordingTool("run_sql")], max_iterations=2).run("Q")

    replayed = model.requests[1][0][2]["tool_calls"][0]
    assert replayed["extra_content"] == {"google": {"thought_signature": "sig"}}
