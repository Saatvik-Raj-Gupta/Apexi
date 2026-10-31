"""Gemini provider boundary reserved for phase 4."""


class GeminiClient:
    """Placeholder that prevents accidental provider use before configuration."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def unavailable_message(self) -> str:
        return "Gemini integration is deferred until the API key is configured."

