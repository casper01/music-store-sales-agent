from types import SimpleNamespace

import httpx
import openai
import pytest

from sales_agent.adapters.chat_model import OpenAICompatibleChatModel, portable_tools
from sales_agent.adapters.embeddings import OpenAICompatibleEmbedder
from sales_agent.errors import EmbeddingError, ModelError


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


def chat_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


NO_ARGS_TOOL = {"type": "function", "function": {"name": "get_schema", "description": "d"}}


def test_chat_model_normalises_tool_calls():
    raw_call = SimpleNamespace(
        id="call-9",
        type="function",
        function=SimpleNamespace(name="run_sql", arguments='{"query": "SELECT 1"}'),
    )
    completions = FakeCompletions(completion(tool_calls=[raw_call]))
    model = OpenAICompatibleChatModel(
        "key", "gpt-test", temperature=0, client=chat_client(completions)
    )

    response = model.complete([{"role": "user", "content": "q"}], [NO_ARGS_TOOL])

    assert response.tool_calls[0].name == "run_sql"
    assert response.tool_calls[0].arguments_json == '{"query": "SELECT 1"}'
    request = completions.requests[0]
    assert request["model"] == "gpt-test"
    assert request["tool_choice"] == "auto"
    assert request["temperature"] == 0


def test_chat_model_omits_optional_parameters():
    completions = FakeCompletions(completion(content="answer"))
    model = OpenAICompatibleChatModel("key", "gpt-test", client=chat_client(completions))

    response = model.complete([], [])

    assert response.content == "answer"
    assert response.tool_calls == ()
    assert "tools" not in completions.requests[0]
    assert "temperature" not in completions.requests[0]


def test_chat_model_translates_sdk_errors():
    request = httpx.Request("POST", "https://api.example.test")
    completions = FakeCompletions(error=openai.APITimeoutError(request=request))
    model = OpenAICompatibleChatModel("key", "gpt-test", client=chat_client(completions))

    with pytest.raises(ModelError, match="timed out"):
        model.complete([], [])


def test_embedder_batches_and_preserves_order():
    class FakeEmbeddings:
        def __init__(self):
            self.batches = []

        def create(self, model, input):
            self.batches.append(list(input))
            data = [
                SimpleNamespace(index=index, embedding=[float(len(text))])
                for index, text in enumerate(input)
            ]
            return SimpleNamespace(data=list(reversed(data)))

    embeddings = FakeEmbeddings()
    embedder = OpenAICompatibleEmbedder(
        "key", "embed-test", batch_size=2, client=SimpleNamespace(embeddings=embeddings)
    )

    vectors = embedder.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert embeddings.batches == [["a", "bb"], ["ccc"]]


def test_embedder_translates_sdk_errors():
    class FailingEmbeddings:
        def create(self, model, input):
            raise openai.OpenAIError("boom")

    embedder = OpenAICompatibleEmbedder(
        "key", "m", client=SimpleNamespace(embeddings=FailingEmbeddings())
    )
    with pytest.raises(EmbeddingError, match="boom"):
        embedder.embed(["text"])


def test_gemini_thought_signature_is_preserved_and_missing_id_is_filled():
    raw_call = SimpleNamespace(
        id="",
        type="function",
        function=SimpleNamespace(name="get_schema", arguments="{}"),
        model_extra={"extra_content": {"google": {"thought_signature": "sig-123"}}},
    )
    completions = FakeCompletions(completion(tool_calls=[raw_call]))
    model = OpenAICompatibleChatModel("key", "gemini-test", client=chat_client(completions))

    call = model.complete([], [])
    [request] = call.tool_calls

    assert request.id == "call_1"
    assert request.provider_fields == {
        "extra_content": {"google": {"thought_signature": "sig-123"}}
    }
    echoed = call.to_message()["tool_calls"][0]
    assert echoed["extra_content"]["google"]["thought_signature"] == "sig-123"
    assert echoed["id"] == "call_1"


def test_tool_schemas_are_reduced_to_a_portable_subset():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 10}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_schema",
                "description": "d",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]

    run_sql, get_schema = portable_tools(tools)

    assert run_sql["function"]["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    assert "parameters" not in get_schema["function"]
    assert tools[0]["function"]["parameters"]["additionalProperties"] is False


def test_unknown_model_gives_actionable_error():
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(404, request=request)
    error = openai.NotFoundError("not found", response=response, body=None)
    model = OpenAICompatibleChatModel(
        "key", "gemini-typo", client=chat_client(FakeCompletions(error=error))
    )

    with pytest.raises(ModelError, match="LLM_MODEL"):
        model.complete([], [])
