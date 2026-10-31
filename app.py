"""Streamlit UI for the bounded Gemini API testing assistant."""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.orchestrator import AgentRunResult, run_agent
from services.llm_client import GeminiClientError
from services.safety import MAX_TOOL_CALLS

load_dotenv()

SAMPLE_TASKS = {
    "Users: status and fields": (
        "Test the demo users endpoint. Confirm HTTP 200 and verify every returned "
        "user has an id and email."
    ),
    "Health: status": "Check the demo health endpoint and confirm HTTP 200.",
    "Failure: expected status": (
        "Test the demo missing endpoint and confirm that it returns HTTP 200."
    ),
}


def get_api_key() -> str | None:
    """Read the key from Streamlit Secrets first, then the local environment."""
    try:
        value = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError, TypeError):
        value = None
    return str(value) if value else os.getenv("GEMINI_API_KEY") or None


def _render_report(result: AgentRunResult) -> None:
    report = result.report.model_dump()
    passed = report["passed"]
    failed = report["failed"]
    total = report["total_tests"]
    columns = st.columns(4)
    columns[0].metric("Overall", report["overall_status"])
    columns[1].metric("Total checks", total)
    columns[2].metric("Passed", passed)
    columns[3].metric("Failed", failed)

    st.subheader("Results")
    if report["results"]:
        st.dataframe(
            [
                {
                    "Status": item["status"],
                    "Endpoint": item["endpoint_id"],
                    "Check": item["check"],
                    "Expected": item["expected"],
                    "Actual": item["actual"],
                    "Evidence": item["evidence"],
                }
                for item in report["results"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No completed checks were recorded.")

    if result.final_text:
        st.subheader("Agent summary")
        st.write(result.final_text)
    with st.expander("Activity timeline", expanded=True):
        for index, event in enumerate(result.context.activity, start=1):
            label = event.get("tool") or event.get("step", "event")
            status = event.get("status", "completed")
            st.write(f"{index}. `{label}` — {status}")
            with st.expander("Details", expanded=False):
                st.json(event)
    with st.expander("Full JSON report"):
        st.json(report)
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="apexi-report.json",
        mime="application/json",
    )


def _render_gemini_error(error: GeminiClientError) -> None:
    st.error(str(error))
    with st.expander("Gemini error details — for debugging", expanded=True):
        st.json(error.diagnostics)
        st.caption(
            "Do not share API keys or other sensitive values from provider "
            "diagnostics. The key itself is redacted by the client."
        )


st.set_page_config(page_title="Apexi API Testing Assistant", page_icon="🧪")
st.title("Apexi API Testing Assistant")
st.write(
    "Describe a safe API check in natural language. The bounded agent uses only "
    "the built-in synthetic demo endpoints."
)
st.warning(
    "Authorized testing only. This demo is not a production security scanner and "
    f"allows at most {MAX_TOOL_CALLS} tool calls per run."
)

api_key_configured = get_api_key() is not None
st.caption(
    "Gemini API key: configured"
    if api_key_configured
    else "Gemini API key: not configured — add GEMINI_API_KEY to .env or Streamlit Secrets."
)

sample_name = st.selectbox("Sample task", ["Custom"] + list(SAMPLE_TASKS))
default_task = "" if sample_name == "Custom" else SAMPLE_TASKS[sample_name]
task = st.text_area(
    "Testing objective",
    value=default_task,
    max_chars=1000,
    placeholder="Example: Check the demo users endpoint for HTTP 200 and email.",
)

if st.button("Run agent", type="primary", disabled=not api_key_configured):
    if not task.strip():
        st.error("Enter a testing objective before running the agent.")
    else:
        with st.spinner("Running bounded Gemini tool calls..."):
            try:
                result = run_agent(task)
            except GeminiClientError as error:
                _render_gemini_error(error)
            except (ValueError, RuntimeError, TypeError) as error:
                st.error(f"Agent run could not complete: {error}")
            else:
                st.session_state["last_result"] = result

last_result: AgentRunResult | None = st.session_state.get("last_result")
if last_result is not None:
    _render_report(last_result)
