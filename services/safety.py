"""Safety policy and bounds for the deterministic first version."""

MAX_TOOL_CALLS = 8
MAX_MODEL_TURNS = MAX_TOOL_CALLS + 1
MAX_PROVIDER_RETRIES = 1
MAX_TASK_LENGTH = 1000
MAX_QUERY_PARAMS = 5
MAX_QUERY_VALUE_LENGTH = 100
MAX_REQUIRED_FIELDS = 10
MAX_RESPONSE_BYTES = 16_384
ALLOWED_METHODS = frozenset({"GET"})
SENSITIVE_KEYS = frozenset({"authorization", "cookie", "set-cookie", "api_key", "token"})


def redact_secrets(value: object) -> object:
    """Recursively redact common secret-bearing mapping keys."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value
