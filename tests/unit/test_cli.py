import pytest

from sales_agent.agent.contracts import AgentResult, StopReason, ToolCallRecord
from sales_agent.cli import EXAMPLE_PROMPTS, SESSION_QUESTIONS, build_parser, main


@pytest.mark.parametrize(
    ("argv", "command"),
    [(["index"], "index"), (["ask", "q"], "ask"), (["eval"], "eval"), (["session"], "session")],
)
def test_parser_accepts_subcommands(argv, command):
    assert build_parser().parse_args(argv).command == command


def test_missing_api_key_exits_with_readable_error(monkeypatch, capsys):
    monkeypatch.setenv("LLM_API_KEY", "")

    exit_code = main(["ask", "Which country has the most revenue?"])

    assert exit_code == 1
    assert "LLM_API_KEY is not set" in capsys.readouterr().err


def test_invalid_configuration_exits_with_readable_error(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "zero")

    assert main(["index"]) == 1
    assert "AGENT_MAX_ITERATIONS" in capsys.readouterr().err


def test_session_uses_five_questions():
    assert len(SESSION_QUESTIONS) == 5


class EchoAgent:
    """Stands in for the LLM agent; answers with the question so output is traceable."""

    def run(self, question):
        call = ToolCallRecord("run_sql", {"query": "SELECT 1"}, ok=True)
        return AgentResult(f"Echo: {question}", StopReason.ANSWERED, 2, (call,))


@pytest.fixture
def offline_agent(monkeypatch, chinook_database):
    monkeypatch.setenv("DATABASE_PATH", str(chinook_database))
    monkeypatch.setattr("sales_agent.cli.build_agent", lambda settings, **_: EchoAgent())


def test_session_command_records_live_output(offline_agent, tmp_path):
    output = tmp_path / "session.md"

    assert main(["session", "--output", str(output)]) == 0

    markdown = output.read_text(encoding="utf-8")
    assert markdown.count("## ") == 5
    assert f"Echo: {SESSION_QUESTIONS[0]}" in markdown
    assert "SELECT 1" in markdown


def test_eval_command_writes_report_and_fails_on_unmet_criteria(offline_agent, tmp_path, capsys):
    output = tmp_path / "report.md"

    exit_code = main(["eval", "--output", str(output)])

    assert exit_code == 1
    assert "database unchanged: True" in capsys.readouterr().out
    assert "# Evaluation report" in output.read_text(encoding="utf-8")


def test_chat_shows_example_prompts_and_answers(offline_agent, monkeypatch, capsys):
    inputs = iter([EXAMPLE_PROMPTS[0][1], "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    assert main(["chat"]) == 0

    out = capsys.readouterr().out
    assert "Example prompts:" in out
    assert all(prompt in out for _, prompt in EXAMPLE_PROMPTS)
    assert f"Echo: {EXAMPLE_PROMPTS[0][1]}" in out
