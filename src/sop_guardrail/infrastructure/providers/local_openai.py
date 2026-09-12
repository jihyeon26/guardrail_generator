"""Adapter for a locally hosted OpenAI-compatible server (LM Studio, Ollama, vLLM)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from time import perf_counter
from typing import Any, Protocol, TypeVar

from pydantic import Field, SecretStr, ValidationError, field_validator

from sop_guardrail.domain.errors import ModelInvocationError
from sop_guardrail.domain.models import (
    ContractModel,
    ModelMetadata,
    ModelResult,
    ModelTask,
)
from sop_guardrail.infrastructure.providers.instructions import SYSTEM_INSTRUCTIONS

OutputT = TypeVar("OutputT", bound=ContractModel)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
# Local reasoning models are slow; a cloud-sized timeout aborts healthy calls.
DEFAULT_TIMEOUT_SECONDS = 600

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ChatCompletionTransport(Protocol):
    """Minimal seam so tests can drive the adapter without a running server."""

    def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class LocalOpenAISettings(ContractModel):
    """Connection details for a local server that speaks /v1/chat/completions."""

    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0, ge=0, le=2)
    disable_thinking: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not normalized.endswith("/v1"):
            raise ValueError("base_url must end with /v1")
        return normalized

    @classmethod
    def from_env(cls) -> LocalOpenAISettings:
        raw_key = os.getenv("LOCAL_LLM_API_KEY")
        timeout = os.getenv("LOCAL_LLM_TIMEOUT_SECONDS")
        return cls(
            base_url=os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ["LOCAL_LLM_MODEL"],
            api_key=SecretStr(raw_key) if raw_key else None,
            timeout_seconds=float(timeout) if timeout else DEFAULT_TIMEOUT_SECONDS,
            disable_thinking=os.getenv("LOCAL_LLM_DISABLE_THINKING", "").lower()
            in {"1", "true", "yes"},
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"


class UrllibTransport:
    """Standard-library HTTP transport; the local adapter adds no dependencies."""

    def __init__(self, *, timeout_seconds: float, api_key: SecretStr | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key

    def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        request = urllib.request.Request(  # noqa: S310 - scheme is validated by the settings
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelInvocationError(f"local model request to {url} failed") from exc
        if not isinstance(body, dict):
            raise ModelInvocationError("local model returned a non-object response")
        return body


class LocalOpenAIGateway:
    """Constrain a local model to the workflow's Pydantic contracts.

    Local reasoning builds often emit the constrained JSON in ``reasoning_content``
    while leaving ``content`` empty, so both fields are inspected. A response that
    still fails contract validation is retried with the validation error attached.
    """

    def __init__(
        self,
        settings: LocalOpenAISettings,
        *,
        transport: ChatCompletionTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport or UrllibTransport(
            timeout_seconds=settings.timeout_seconds,
            api_key=settings.api_key,
        )

    def invoke(
        self,
        *,
        task: ModelTask,
        prompt: str,
        response_model: type[OutputT],
    ) -> ModelResult[OutputT]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS[task]},
            {"role": "user", "content": prompt},
        ]
        started = perf_counter()
        last_error: str = "no attempt was made"

        for _ in range(self.settings.max_retries + 1):
            body = self._transport.post(
                self.settings.chat_completions_url,
                self._request_payload(messages, response_model),
            )
            text = _structured_text(body)
            try:
                output = response_model.model_validate_json(_json_object(text))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                messages = [*messages, *_repair_messages(text, last_error)]
                continue
            return ModelResult[OutputT](
                output=output,
                metadata=ModelMetadata(
                    provider="local_openai",
                    model=str(body.get("model") or self.settings.model),
                    request_id=str(body["id"]) if body.get("id") else None,
                    latency_ms=round((perf_counter() - started) * 1000),
                ),
            )

        raise ModelInvocationError(
            f"local model returned invalid structured output for {task}: {last_error}"
        )

    def _request_payload(
        self, messages: list[dict[str, str]], response_model: type[OutputT]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        if self.settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload


def _structured_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelInvocationError("local model returned no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ModelInvocationError("local model returned no message")

    for key in ("content", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_wrappers(value)
    raise ModelInvocationError("local model returned an empty message")


def _strip_wrappers(text: str) -> str:
    return _CODE_FENCE.sub("", _THINK_BLOCK.sub("", text)).strip()


def _json_object(text: str) -> str:
    """Return the first JSON object in the text, ignoring any surrounding prose."""

    start = text.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return text[start:end]


def _repair_messages(previous: str, error: str) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": previous},
        {
            "role": "user",
            "content": (
                "The previous response did not satisfy the required schema:\n"
                f"{error}\n\nReturn corrected JSON only."
            ),
        },
    ]
