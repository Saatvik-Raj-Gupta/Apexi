"""Safe, deterministic tools backed by an in-app synthetic API."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent.schemas import (
    GenerateReportInput,
    ListEndpointsInput,
    RunApiTestInput,
    ValidateResponseInput,
)
from services.reporting import build_report
from services.safety import ALLOWED_METHODS, MAX_RESPONSE_BYTES, redact_secrets

if TYPE_CHECKING:
    from agent.orchestrator import RunContext

ENDPOINTS: dict[str, dict[str, Any]] = {
    "users_list": {
        "method": "GET",
        "path": "/demo/users",
        "checks": ["HTTP status", "required JSON fields"],
        "status": 200,
        "body": {"users": [{"id": 1, "email": "ada@example.test"}, {"id": 2, "email": "linus@example.test"}]},
    },
    "health": {
        "method": "GET",
        "path": "/demo/health",
        "checks": ["HTTP status", "required JSON fields"],
        "status": 200,
        "body": {"status": "ok"},
    },
    "missing": {
        "method": "GET",
        "path": "/demo/missing",
        "checks": ["HTTP status"],
        "status": 404,
        "body": {"error": "not found"},
    },
}


def _record_activity(context: RunContext, tool: str, args: Any, result: Any) -> None:
    context.activity.append(
        {
            "step": "tool",
            "tool": tool,
            "status": "success",
            "arguments": redact_secrets(args),
            "result": redact_secrets(result),
        }
    )


def list_demo_endpoints(
    context: RunContext, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    context.consume_tool_call()
    ListEndpointsInput.model_validate(arguments or {})
    result = {
        "endpoints": [
            {
                "endpoint_id": endpoint_id,
                "method": item["method"],
                "path": item["path"],
                "checks": item["checks"],
            }
            for endpoint_id, item in ENDPOINTS.items()
        ]
    }
    _record_activity(context, "list_demo_endpoints", arguments or {}, result)
    return result


def run_api_test(context: RunContext, arguments: dict[str, Any]) -> dict[str, Any]:
    context.consume_tool_call()
    parsed = RunApiTestInput.model_validate(arguments)
    if parsed.method not in ALLOWED_METHODS:
        raise ValueError(f"Method {parsed.method} is not allowed.")
    endpoint = ENDPOINTS.get(parsed.endpoint_id)
    if endpoint is None:
        raise ValueError(f"Unknown demo endpoint: {parsed.endpoint_id}.")
    started = time.perf_counter()
    body = endpoint["body"]
    encoded = json.dumps(body).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeds the configured size limit.")
    test_id = uuid4().hex
    result = {
        "test_id": test_id,
        "endpoint_id": parsed.endpoint_id,
        "method": parsed.method,
        "status_code": endpoint["status"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "body": body,
        "expected_status": parsed.expected_status,
        "required_json_fields": parsed.required_json_fields,
    }
    result["checks"] = _validate_captured_response(result)
    context.tests.append(result)
    _record_activity(context, "run_api_test", arguments, result)
    return redact_secrets(result)


def _has_field(value: Any, field: str) -> bool:
    if isinstance(value, dict):
        if field in value:
            return True
        return any(_has_field(item, field) for item in value.values())
    if isinstance(value, list):
        return all(_has_field(item, field) for item in value)
    return False


def _validate_captured_response(captured: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {
            "endpoint_id": captured["endpoint_id"],
            "check": "HTTP status",
            "status": "PASS"
            if captured["status_code"] == captured["expected_status"]
            else "FAIL",
            "expected": str(captured["expected_status"]),
            "actual": str(captured["status_code"]),
            "evidence": "Observed synthetic HTTP response status.",
        }
    ]
    for field in captured["required_json_fields"]:
        present = _has_field(captured["body"], field)
        checks.append(
            {
                "endpoint_id": captured["endpoint_id"],
                "check": f"JSON field: {field}",
                "status": "PASS" if present else "FAIL",
                "expected": field,
                "actual": "present" if present else "missing",
                "evidence": "Checked the captured JSON response in Python.",
            }
        )
    return checks


def validate_response(context: RunContext, arguments: dict[str, Any]) -> dict[str, Any]:
    context.consume_tool_call()
    parsed = ValidateResponseInput.model_validate(arguments)
    captured = next((item for item in context.tests if item["test_id"] == parsed.test_id), None)
    if captured is None:
        raise ValueError("Unknown test ID.")
    captured["required_json_fields"] = parsed.required_json_fields
    checks = _validate_captured_response(captured)
    captured["checks"] = checks
    _record_activity(context, "validate_response", arguments, {"checks": checks})
    return {"test_id": parsed.test_id, "checks": checks}


def generate_report(
    context: RunContext, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    context.consume_tool_call()
    parsed = GenerateReportInput.model_validate(arguments or {})
    task = parsed.task or context.task
    records = [check for test in context.tests for check in test.get("checks", [])]
    result = build_report(task, records).model_dump()
    _record_activity(context, "generate_report", arguments or {}, result)
    return result
