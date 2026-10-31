from agent.orchestrator import RunContext


def test_context_tracks_tool_budget_and_activity() -> None:
    context = RunContext("demo")
    context.consume_tool_call()
    context.activity.append({"tool": "example"})
    assert context.tool_calls == 1
    assert context.activity[0]["tool"] == "example"
