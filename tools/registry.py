"""Allowlisted tool registry and dispatcher."""

from collections.abc import Callable
from typing import Any

from agent.orchestrator import RunContext
from tools.api_tools import generate_report, list_demo_endpoints, run_api_test, validate_response

TOOLS: dict[str, Callable[[RunContext, dict[str, Any]], dict[str, Any]]] = {
    "list_demo_endpoints": list_demo_endpoints,
    "run_api_test": run_api_test,
    "validate_response": validate_response,
    "generate_report": generate_report,
}


def dispatch_tool(context: RunContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}.")
    return tool(context, arguments)

