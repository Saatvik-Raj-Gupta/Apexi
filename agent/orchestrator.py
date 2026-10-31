"""Bounded model/tool orchestration."""

from dataclasses import dataclass, field
from typing import Any

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import Report, RunTaskInput
from services.llm_client import GeminiClient, GeminiClientError
from services.reporting import build_report
from services.safety import MAX_MODEL_TURNS, MAX_TOOL_CALLS, redact_secrets
from tools.registry import TOOL_DECLARATIONS, dispatch_tool


@dataclass
class RunContext:
    """State shared by deterministic tools during one run."""

    task: str
    tool_calls: int = 0
    tests: list[dict[str, Any]] = field(default_factory=list)
    activity: list[dict[str, Any]] = field(default_factory=list)

    def consume_tool_call(self) -> None:
        if self.tool_calls >= MAX_TOOL_CALLS:
            raise RuntimeError(f"Tool-call budget exceeded ({MAX_TOOL_CALLS}).")
        self.tool_calls += 1


@dataclass(frozen=True)
class AgentRunResult:
    report: Report
    final_text: str
    context: RunContext


def _record_model_activity(context: RunContext, response: Any) -> None:
    context.activity.append(
        {
            "step": "model",
            "tool_calls": [
                {"name": call.name, "arguments": redact_secrets(call.arguments)}
                for call in response.function_calls
            ],
            "text": response.text[:1000],
            "duration_ms": response.duration_ms,
            "attempts": response.attempts,
        }
    )


def run_agent(
    task: str,
    client: GeminiClient | None = None,
    context: RunContext | None = None,
) -> AgentRunResult:
    """Run sequential Gemini tool calls and return a trusted deterministic report."""
    parsed_task = RunTaskInput(task=task).task
    run_context = context or RunContext(task=parsed_task)
    if run_context.task != parsed_task:
        raise ValueError("Run context task does not match the requested task.")
    provider = client or GeminiClient()
    messages: list[dict[str, Any]] = [{"role": "user", "content": parsed_task}]
    final_text = ""

    for _ in range(MAX_MODEL_TURNS):
        try:
            response = provider.generate(messages, TOOL_DECLARATIONS, SYSTEM_PROMPT)
        except GeminiClientError as exc:
            run_context.activity.append(
                {"step": "model", "status": "error", "error": str(exc)}
            )
            raise
        _record_model_activity(run_context, response)
        if not response.function_calls:
            final_text = response.text
            break
        messages.append(
            {
                "role": "model",
                "content": response.text,
                "model_parts": response.model_parts,
                "function_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in response.function_calls
                ],
            }
        )
        for call in response.function_calls:
            try:
                result = dispatch_tool(run_context, call.name, call.arguments)
                tool_message = {"name": call.name, "response": result}
            except (ValueError, RuntimeError, TypeError) as exc:
                result = {"error": str(exc)}
                tool_message = {"name": call.name, "response": result}
                run_context.activity.append(
                    {
                        "step": "tool",
                        "tool": call.name,
                        "status": "error",
                        "arguments": redact_secrets(call.arguments),
                        "error": str(exc),
                    }
                )
                if isinstance(exc, RuntimeError) and run_context.tool_calls >= MAX_TOOL_CALLS:
                    raise
            messages.append({"role": "tool", **tool_message})
    else:
        run_context.activity.append(
            {"step": "orchestrator", "status": "error", "error": "Model turn budget exceeded."}
        )
        raise RuntimeError(f"Model-turn budget exceeded ({MAX_MODEL_TURNS}).")

    records = [check for test in run_context.tests for check in test.get("checks", [])]
    return AgentRunResult(
        report=build_report(parsed_task, records),
        final_text=final_text,
        context=run_context,
    )
