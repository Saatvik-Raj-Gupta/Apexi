# Apexi — Agentic API Testing Assistant

A safe API-testing assistant. It runs bounded Gemini function calls against an
in-app synthetic API and computes all test results deterministically in Python.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Run the automated checks with:

```bash
pytest -q
```

## Current scope

- Four registered Python tools: endpoint discovery, API test execution,
  response validation, and deterministic report generation.
- Only fixed synthetic endpoints and `GET` are allowed.
- Strict Pydantic schemas reject unknown fields and malformed arguments.
- Query parameters, required fields, response size, task length, and tool calls
  are bounded.
- Report counts and status are computed from captured evidence in Python.
- Secrets are redacted from activity records.
- `run_agent()` supports bounded sequential Gemini tool calls and preserves a
  model/tool activity timeline.
- Gemini provider errors and missing API keys are surfaced as friendly errors;
  they never become successful test results.
- The Streamlit UI keeps the last completed result in session state, so widget
  reruns do not automatically repeat an agent run.

The app does not accept arbitrary URLs, credentials, shell commands, or
destructive methods. Use only with systems you are authorized to test.

## Gemini agent usage

The Gemini integration uses the current `google-genai` SDK. Configure
`GEMINI_API_KEY` in `.env` locally or in Streamlit Secrets, then call the
orchestrator from the application layer:

```python
from agent.orchestrator import run_agent

result = run_agent("Check the demo users endpoint for HTTP 200 and user emails.")
print(result.report.model_dump())
```

The orchestrator sends only the registered tool declarations, validates every
tool call before dispatch, stops at the eight-call budget, and computes report
counts in Python from captured evidence. The UI displays the activity timeline,
result table, metrics, validated JSON, and a downloadable report.

## Evaluation coverage

The automated evaluation suite covers:

- Valid endpoint and required-field checks.
- Wrong expected status and missing fields.
- Unknown endpoints, extra arguments, and unregistered tools.
- Tool-call budget exhaustion.
- Invalid final report counts.
- Provider errors and missing API keys.

Run the complete suite with `pytest -q`; evaluation results are measured by the
tests and are not represented as an invented quality score.

## Streamlit URLs

When Streamlit starts locally, it may print three addresses:

- **Local URL** (`localhost` or `127.0.0.1`): use this on the same computer.
- **Network URL** (a private address such as `192.168.x.x`): use this from
  another device on the same Wi-Fi/LAN, if the firewall allows it.
- **External URL**: an address that may be reachable outside the local network.
  Do not share or expose it unless you intentionally configured firewall,
  authentication, and network access. For normal development, use the Local
  URL.

If Gemini fails, the app shows a friendly error and an expandable
**Gemini error details** panel containing a safe error code, provider message,
exception type, and HTTP status when the SDK provides one. API keys are
redacted.

The Gemini tool declarations are converted to `google-genai` `Schema` objects
before requests are sent. The first version intentionally does not advertise
arbitrary query parameters to Gemini because the Google function-calling schema
does not accept `additionalProperties` for open-ended object fields.

Set `GEMINI_MODEL` in `.env` to select the model. If it is omitted, the client
uses `gemini-2.5-flash`.

Function results are converted to Gemini-compatible user content parts before
being sent. The internal orchestrator may label these messages as `tool`, but
the provider adapter never sends `role="tool"` to Gemini because that is not a
valid native Gemini content role.

The provider adapter also retains the SDK's complete model parts between turns,
including any `thoughtSignature`, instead of rebuilding function-call parts
from only the function name and arguments. This is required by newer Gemini
models during sequential function calling.
