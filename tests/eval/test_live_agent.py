"""Live evaluation against the configured LLM.

Skipped unless LLM_API_KEY is set. Run with:  pytest -m live
The index must exist (``sales-agent index``). Each case runs against a private copy of the
database, so the shipped file is never at risk even in this test.
"""

import os
import shutil

import pytest

from sales_agent.config import Settings
from sales_agent.evaluation.cases import load_cases
from sales_agent.evaluation.evaluator import Evaluator, file_sha256
from sales_agent.factory import build_agent, build_reader

from conftest import CASES_PATH, CHINOOK_PATH

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("LLM_API_KEY"), reason="LLM_API_KEY is not set"),
]


@pytest.fixture(scope="module")
def evaluator(tmp_path_factory):
    database = tmp_path_factory.mktemp("live") / "chinook.sqlite"
    shutil.copyfile(CHINOOK_PATH, database)
    settings = Settings.from_env({**os.environ, "DATABASE_PATH": str(database)})
    return Evaluator(build_agent(settings), build_reader(settings)), database


@pytest.mark.parametrize("case", load_cases(CASES_PATH), ids=lambda case: f"case-{case.id}")
def test_live_case(evaluator, case):
    runner, database = evaluator
    checksum = file_sha256(database)

    result = runner.evaluate_case(case)

    assert file_sha256(database) == checksum
    assert result.passed, f"{result.failures}\n\nTools: {result.tool_names}\n\n{result.answer}"
