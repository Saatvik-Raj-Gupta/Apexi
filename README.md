# Apexi — Agentic API Testing Assistant

A safe first version of an API-testing assistant. It currently runs deterministic
checks against an in-app synthetic API; Gemini function calling is deliberately
reserved for the next implementation phase.

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

The app does not accept arbitrary URLs, credentials, shell commands, or
destructive methods. Use only with systems you are authorized to test.
