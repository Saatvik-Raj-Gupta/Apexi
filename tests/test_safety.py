import pytest

from agent.orchestrator import RunContext
from services.safety import MAX_TOOL_CALLS, redact_secrets
from tools.api_tools import list_demo_endpoints, run_api_test


def test_redacts_sensitive_values() -> None:
    assert redact_secrets({"Authorization": "secret", "nested": {"token": "x"}}) == {
        "Authorization": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }


def test_oversized_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_api_test(
            RunContext("x"),
            {"endpoint_id": "health", "expected_status": 200, "query_params": {"q": "x" * 101}},
        )


def test_tool_budget_is_enforced() -> None:
    context = RunContext("x")
    for _ in range(MAX_TOOL_CALLS):
        list_demo_endpoints(context)
    with pytest.raises(RuntimeError):
        list_demo_endpoints(context)
