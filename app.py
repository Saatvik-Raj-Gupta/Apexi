"""Streamlit entry point for the safe deterministic first version."""

import json
import os

import streamlit as st

from agent.orchestrator import RunContext
from tools.api_tools import generate_report, run_api_test, validate_response


def get_api_key() -> str | None:
    try:
        value = st.secrets.get("GEMINI_API_KEY")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or None


st.set_page_config(page_title="Apexi API Testing Assistant", page_icon="🧪")
st.title("Apexi API Testing Assistant")
st.write("Run evidence-based checks against safe, synthetic demo endpoints.")
st.info("Authorized testing only. Gemini agent integration is intentionally deferred.")

task = st.text_area(
    "Testing objective",
    value="Check the demo users endpoint and verify HTTP 200, id, and email.",
    max_chars=1000,
)
st.caption("Gemini API key: configured" if get_api_key() else "Gemini API key: not configured")

if st.button("Run deterministic demo", type="primary"):
    context = RunContext(task=task)
    try:
        tested = run_api_test(
            context,
            {
                "endpoint_id": "users_list",
                "method": "GET",
                "expected_status": 200,
                "required_json_fields": ["id", "email"],
            },
        )
        validate_response(
            context,
            {"test_id": tested["test_id"], "required_json_fields": ["id", "email"]},
        )
        report = generate_report(context, {"task": task})
        st.metric("Overall status", report["overall_status"])
        st.json(report)
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="apexi-report.json",
            mime="application/json",
        )
        with st.expander("Activity timeline"):
            st.json(context.activity)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
