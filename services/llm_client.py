"""Provider-specific Gemini function-calling adapter."""

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from services.safety import MAX_PROVIDER_RETRIES

load_dotenv()


class GeminiClientError(RuntimeError):
    """A user-safe provider error with safe diagnostics for debugging."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        provider_message: str | None = None,
        status_code: int | str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider_message = provider_message or message
        self.status_code = status_code
        self.error_type = error_type or self.__class__.__name__

    @property
    def diagnostics(self) -> dict[str, str]:
        details = {
            "error_code": self.code,
            "message": self.provider_message,
            "error_type": self.error_type,
        }
        if self.status_code is not None:
            details["status_code"] = str(self.status_code)
        return details


def _safe_provider_message(error: BaseException, api_key: str | None) -> str:
    message = str(error).strip() or "The provider returned no diagnostic message."
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(r"(?i)(key|token|authorization)\s*[=:]\s*[^,\s]+", r"\1=[REDACTED]", message)


def _status_code(error: BaseException) -> int | str | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, (int, str)) and value:
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, (int, str)) and value else None


def _is_retryable_status(value: int | str | None) -> bool:
    try:
        status = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return False
    return status == 429 or status >= 500


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)
    model_parts: list[Any] = field(default_factory=list)
    duration_ms: float = 0.0
    attempts: int = 1


class GeminiClient:
    """Small normalized wrapper around the current google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite"
        self._sdk_client = sdk_client

    def _client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self.api_key:
            raise GeminiClientError(
                "Gemini API key is not configured (GEMINI_API_KEY).",
                code="missing_api_key",
                provider_message=(
                    "GEMINI_API_KEY is not configured. Add it to .env or Streamlit Secrets."
                ),
            )
        try:
            from google import genai

            self._sdk_client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise GeminiClientError(
                "Gemini client initialization failed.",
                code="client_initialization_error",
                provider_message=_safe_provider_message(exc, self.api_key),
                status_code=_status_code(exc),
                error_type=type(exc).__name__,
            ) from exc
        return self._sdk_client

    @staticmethod
    def _to_sdk_contents(messages: list[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message["content"])]))
            elif role == "model":
                if message.get("model_parts"):
                    contents.append(
                        types.Content(role="model", parts=message["model_parts"])
                    )
                    continue
                parts = []
                if message.get("content"):
                    parts.append(types.Part.from_text(text=message["content"]))
                for call in message.get("function_calls", []):
                    parts.append(
                        types.Part.from_function_call(
                            name=call["name"], args=call["arguments"]
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(
                    types.Content(
                        # Gemini represents a function response as a user content
                        # part; "tool" is not a valid Gemini Content role.
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=message["name"], response={"result": message["response"]}
                            )
                        ],
                    )
                )
        return contents

    @staticmethod
    def _to_sdk_declarations(declarations: list[dict[str, Any]]) -> list[Any]:
        """Build SDK Schema objects so JSON Schema keys serialize correctly."""
        from google.genai import types

        def schema(value: dict[str, Any]) -> Any:
            converted = dict(value)
            if "properties" in converted:
                converted["properties"] = {
                    name: schema(item)
                    for name, item in converted["properties"].items()
                }
            if isinstance(converted.get("items"), dict):
                converted["items"] = schema(converted["items"])
            return types.Schema(**converted)

        return [
            types.FunctionDeclaration(
                name=declaration["name"],
                description=declaration.get("description"),
                parameters=schema(declaration["parameters"]),
            )
            for declaration in declarations
        ]

    def generate(
        self,
        messages: list[dict[str, Any]],
        tool_declarations: list[dict[str, Any]],
        system_instruction: str,
    ) -> ModelResponse:
        from google.genai import types

        started = time.perf_counter()
        attempts = 0
        response = None
        while attempts <= MAX_PROVIDER_RETRIES:
            attempts += 1
            try:
                response = self._client().models.generate_content(
                    model=self.model,
                    contents=self._to_sdk_contents(messages),
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[
                            types.Tool(
                                function_declarations=self._to_sdk_declarations(
                                    tool_declarations
                                )
                            )
                        ],
                    ),
                )
                break
            except GeminiClientError:
                raise
            except Exception as exc:
                status_code = _status_code(exc)
                if _is_retryable_status(status_code) and attempts <= MAX_PROVIDER_RETRIES:
                    time.sleep(0.5)
                    continue
                raise GeminiClientError(
                    "Gemini request failed. The provider may be unavailable or rate-limited.",
                    code="provider_request_error",
                    provider_message=_safe_provider_message(exc, self.api_key),
                    status_code=status_code,
                    error_type=type(exc).__name__,
                ) from exc

        calls: list[FunctionCall] = []
        text_parts: list[str] = []
        model_parts: list[Any] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                model_parts.append(part)
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    name = getattr(function_call, "name", "")
                    args = getattr(function_call, "args", {}) or {}
                    calls.append(FunctionCall(name=name, arguments=dict(args)))
                part_text = getattr(part, "text", None)
                if part_text:
                    text_parts.append(part_text)
        return ModelResponse(
            text="".join(text_parts).strip(),
            function_calls=calls,
            model_parts=model_parts,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            attempts=attempts,
        )
