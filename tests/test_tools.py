import pytest

from agent.orchestrator import RunContext
from tools.api_tools import generate_report, list_demo_endpoints, run_api_test, validate_response
from tools.registry import dispatch_tool


def test_list_and_run_and_validate_and_report() -> None:
    context = RunContext("check users")
    endpoints = list_demo_endpoints(context)
    assert {item["endpoint_id"] for item in endpoints["endpoints"]} >= {"users_list", "health"}
    captured = run_api_test(
        context, {"endpoint_id": "users_list", "method": "GET", "expected_status": 200}
    )
    assert all(check["status"] == "PASS" for check in captured["checks"])
    checks = validate_response(
        context, {"test_id": captured["test_id"], "required_json_fields": ["id", "email"]}
    )
    assert all(check["status"] == "PASS" for check in checks["checks"])
    report = generate_report(context)
    assert (report["total_tests"], report["passed"], report["failed"]) == (3, 3, 0)
    assert report["overall_status"] == "PASS"


def test_unknown_endpoint_and_tool_are_rejected() -> None:
    context = RunContext("bad")
    with pytest.raises(ValueError):
        run_api_test(context, {"endpoint_id": "external", "expected_status": 200})
    with pytest.raises(ValueError):
        dispatch_tool(context, "shell", {})


def test_wrong_status_and_missing_field_fail() -> None:
    context = RunContext("failure")
    captured = run_api_test(
        context,
        {"endpoint_id": "health", "expected_status": 201, "required_json_fields": ["missing"]},
    )
    result = validate_response(
        context, {"test_id": captured["test_id"], "required_json_fields": ["missing"]}
    )
    assert [item["status"] for item in result["checks"]] == ["FAIL", "FAIL"]
    assert generate_report(context)["overall_status"] == "FAIL"
