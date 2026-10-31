import pytest
from pydantic import ValidationError

from agent.schemas import Report, RunApiTestInput


def test_tool_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunApiTestInput(endpoint_id="health", expected_status=200, unexpected="nope")


def test_tool_schema_rejects_unsupported_method() -> None:
    with pytest.raises(ValidationError):
        RunApiTestInput(endpoint_id="health", method="POST", expected_status=200)


def test_report_schema_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        Report(
            task="x",
            overall_status="PASS",
            summary="x",
            total_tests=-1,
            passed=0,
            failed=0,
            results=[],
        )
