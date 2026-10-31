"""Bounded orchestration interfaces."""

from dataclasses import dataclass, field
from typing import Any

from services.safety import MAX_TOOL_CALLS


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

