import pytest

from agent.orchestrator import run_agent
from agent.schemas import Report
from services.llm_client import (
    FunctionCall,
    GeminiClient,
    GeminiClientError,
    ModelResponse,
)
from tools.api_tools import run_api_test, validate_response
from tools.registry import dispatch_tool
from agent.orchestrator import RunContext
from tools.registry import TOOL_DECLARATIONS


class ScriptedClient:
    def __init__(self, calls: list[FunctionCall | ModelResponse]) -> None:
        self.calls = iter(calls)

    def generate(self, messages, tool_declarations, system_instruction):
        response = next(self.calls)
        return response if isinstance(response, ModelResponse) else ModelResponse(function_calls=[response])


def run_scripted(task: str, *calls: FunctionCall | ModelResponse):
    return run_agent(task, client=ScriptedClient(list(calls)))


def test_valid_endpoint_and_fields() -> None:
    context = RunContext("Check users.")
    captured = run_api_test(
        context, {"endpoint_id": "users_list", "expected_status": 200}
    )
    result = validate_response(
        context,
        {"test_id": captured["test_id"], "required_json_fields": ["id", "email"]},
    )
    assert all(check["status"] == "PASS" for check in result["checks"])


def test_wrong_status_and_missing_field_are_failures() -> None:
    context = RunContext("failure")
    captured = run_api_test(context, {"endpoint_id": "health", "expected_status": 201})
    checks = validate_response(
        context, {"test_id": captured["test_id"], "required_json_fields": ["missing"]}
    )
    assert [check["status"] for check in checks["checks"]] == ["FAIL", "FAIL"]


def test_unknown_endpoint_and_extra_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        run_api_test(RunContext("x"), {"endpoint_id": "unknown", "expected_status": 200})
    with pytest.raises(ValueError):
        run_api_test(
            RunContext("x"),
            {"endpoint_id": "health", "expected_status": 200, "extra": True},
        )


def test_unregistered_tool_attempt_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool(RunContext("x"), "not_registered", {})


def test_tool_call_budget_is_enforced() -> None:
    class Endless:
        def generate(self, messages, tool_declarations, system_instruction):
            return ModelResponse(function_calls=[FunctionCall("list_demo_endpoints", {})])

    with pytest.raises(RuntimeError, match="Tool-call budget exceeded"):
        run_agent("Repeat forever.", client=Endless())


def test_invalid_final_report_cannot_be_constructed() -> None:
    with pytest.raises(ValueError):
        Report.model_validate(
            {
                "task": "x",
                "overall_status": "PASS",
                "summary": "invalid",
                "total_tests": 1,
                "passed": 1,
                "failed": 0,
                "results": [],
            }
        )


def test_missing_key_produces_friendly_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiClientError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="", sdk_client=None)._client()


def test_provider_error_preserves_safe_debug_details() -> None:
    class ProviderError(Exception):
        status_code = 429

    class Models:
        def generate_content(self, **kwargs):
            raise ProviderError("quota exceeded for key=secret-value")

    class SDK:
        models = Models()

    with pytest.raises(GeminiClientError) as caught:
        GeminiClient(api_key="secret-value", sdk_client=SDK()).generate([], [], "system")

    error = caught.value
    assert str(error) == (
        "Gemini request failed. The provider may be unavailable or rate-limited."
    )
    assert error.diagnostics == {
        "error_code": "provider_request_error",
        "message": "quota exceeded for key=[REDACTED]",
        "error_type": "ProviderError",
        "status_code": "429",
    }


def test_tool_declarations_use_sdk_schema_without_snake_case_fields() -> None:
    declarations = GeminiClient._to_sdk_declarations(TOOL_DECLARATIONS)
    serialized = [
        declaration.model_dump(by_alias=True, exclude_none=True)
        for declaration in declarations
    ]
    assert "additionalProperties" not in str(serialized)
    assert "additional_properties" not in str(serialized)


def test_model_is_loaded_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "configured-model")
    assert GeminiClient(api_key="test").model == "configured-model"


def test_default_model_is_lite(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert GeminiClient(api_key="test").model == "gemini-3.1-flash-lite"


def test_transient_provider_error_is_retried_once(monkeypatch) -> None:
    monkeypatch.setattr("services.llm_client.time.sleep", lambda _: None)

    class ProviderError(Exception):
        status_code = 503

    class Models:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("temporarily unavailable")
            return type("Response", (), {"candidates": []})()

    models = Models()

    class SDK:
        pass

    sdk = SDK()
    sdk.models = models
    response = GeminiClient(api_key="test", sdk_client=sdk).generate([], [], "system")
    assert response.attempts == 2
    assert models.calls == 2


def test_function_responses_use_gemini_user_role() -> None:
    contents = GeminiClient._to_sdk_contents(
        [
            {"role": "user", "content": "Check health."},
            {
                "role": "model",
                "content": "",
                "function_calls": [
                    {"name": "run_api_test", "arguments": {"endpoint_id": "health"}}
                ],
            },
            {
                "role": "tool",
                "name": "run_api_test",
                "response": {"status_code": 200},
            },
        ]
    )
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[2].parts[0].function_response.name == "run_api_test"


def test_model_thought_signature_is_preserved_in_follow_up_content() -> None:
    from google.genai import types

    original_parts = [
        types.Part(
            function_call=types.FunctionCall(
                name="list_demo_endpoints", args={}
            ),
            thought_signature=b"opaque-signature",
        )
    ]
    contents = GeminiClient._to_sdk_contents(
        [{"role": "model", "model_parts": original_parts}]
    )
    assert contents[0].parts[0].thought_signature == b"opaque-signature"
    assert contents[0].parts[0].function_call.name == "list_demo_endpoints"
