import pytest

from agent.orchestrator import RunContext, run_agent
from services.llm_client import FunctionCall, GeminiClientError, ModelResponse


def test_context_tracks_tool_budget_and_activity() -> None:
    context = RunContext("demo")
    context.consume_tool_call()
    context.activity.append({"tool": "example"})
    assert context.tool_calls == 1
    assert context.activity[0]["tool"] == "example"


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, tool_declarations, system_instruction):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                function_calls=[FunctionCall("list_demo_endpoints", {})]
            )
        if self.calls == 2:
            return ModelResponse(
                function_calls=[
                    FunctionCall(
                        "run_api_test",
                        {
                            "endpoint_id": "health",
                            "method": "GET",
                            "expected_status": 200,
                        },
                    )
                ]
            )
        if self.calls == 3:
            test_id = messages[-1]["response"]["test_id"]
            return ModelResponse(
                function_calls=[
                    FunctionCall(
                        "validate_response",
                        {"test_id": test_id, "required_json_fields": ["status"]},
                    )
                ]
            )
        return ModelResponse(text="The health check passed.")


def test_agent_supports_sequential_tools_and_trusted_report() -> None:
    result = run_agent("Check the health endpoint.", client=FakeClient())
    assert result.report.overall_status == "PASS"
    assert result.report.passed == 2
    assert [item["tool"] for item in result.context.activity if item["step"] == "tool"] == [
        "list_demo_endpoints",
        "run_api_test",
        "validate_response",
    ]


class UnknownToolClient:
    def generate(self, messages, tool_declarations, system_instruction):
        if len(messages) == 1:
            return ModelResponse(function_calls=[FunctionCall("not_registered", {})])
        return ModelResponse(text="Finished.")


def test_unknown_tool_is_reported_and_loop_continues() -> None:
    result = run_agent("Try an unknown tool.", client=UnknownToolClient())
    errors = [item for item in result.context.activity if item.get("status") == "error"]
    assert errors[0]["tool"] == "not_registered"
    assert result.report.overall_status == "ERROR"


class FailingClient:
    def generate(self, messages, tool_declarations, system_instruction):
        raise GeminiClientError("provider unavailable")


def test_provider_error_is_preserved_in_activity() -> None:
    with pytest.raises(GeminiClientError):
        run_agent("Check health.", client=FailingClient())


def test_tool_budget_error_is_not_silently_reported() -> None:
    class EndlessClient:
        def generate(self, messages, tool_declarations, system_instruction):
            return ModelResponse(function_calls=[FunctionCall("list_demo_endpoints", {})])

    with pytest.raises(RuntimeError, match="Tool-call budget exceeded"):
        run_agent("Keep listing.", client=EndlessClient())
