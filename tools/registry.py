"""Allowlisted tool registry and dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tools.api_tools import generate_report, list_demo_endpoints, run_api_test, validate_response

if TYPE_CHECKING:
    from agent.orchestrator import RunContext

TOOLS: dict[str, Callable[[RunContext, dict[str, Any]], dict[str, Any]]] = {
    "list_demo_endpoints": list_demo_endpoints,
    "run_api_test": run_api_test,
    "validate_response": validate_response,
    "generate_report": generate_report,
}

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "list_demo_endpoints",
        "description": "List the fixed synthetic API endpoints available for safe testing.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "run_api_test",
        "description": "Run one approved GET request against a fixed synthetic endpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint_id": {"type": "string"},
                "method": {"type": "string", "enum": ["GET"]},
                "expected_status": {"type": "integer", "minimum": 100, "maximum": 599},
                "required_json_fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["endpoint_id", "expected_status"],
        },
    },
    {
        "name": "validate_response",
        "description": "Deterministically validate a captured response and required JSON fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "test_id": {"type": "string"},
                "required_json_fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["test_id"],
        },
    },
    {
        "name": "generate_report",
        "description": "Create a report whose counts are computed from captured test evidence.",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
        },
    },
]


def dispatch_tool(context: RunContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}.")
    return tool(context, arguments)
