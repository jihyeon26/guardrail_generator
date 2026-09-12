from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from sop_guardrail.domain.errors import ModelInvocationError
from sop_guardrail.domain.models import LLMAssessment, ModelTask
from sop_guardrail.infrastructure.providers.local_openai import (
    LocalOpenAIGateway,
    LocalOpenAISettings,
)
from tests.helpers import passing_assessment


class _RecordingTransport:
    """Return queued server bodies and retain the requests for assertions."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = list(bodies)
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.urls.append(url)
        self.requests.append(payload)
        return self._bodies.pop(0)


def _body(
    *, content: str | None = None, reasoning: str | None = None, **extra: Any
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-local-1",
        "model": "qwen-test",
        "choices": [{"message": message}],
        **extra,
    }


def _settings(**overrides: Any) -> LocalOpenAISettings:
    return LocalOpenAISettings(model="qwen-test", **overrides)


def _assessment_json() -> str:
    return passing_assessment().model_dump_json()


def test_settings_default_to_the_local_server_and_normalize_the_base_url() -> None:
    assert _settings().base_url == "http://127.0.0.1:1234/v1"
    assert _settings(base_url="http://127.0.0.1:1234/v1/").chat_completions_url == (
        "http://127.0.0.1:1234/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "base_url", ["127.0.0.1:1234/v1", "http://127.0.0.1:1234", "http://127.0.0.1:1234/openai"]
)
def test_settings_reject_an_unusable_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        _settings(base_url=base_url)


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen-env")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "test-only")
    monkeypatch.setenv("LOCAL_LLM_DISABLE_THINKING", "true")
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_SECONDS", "42")
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)

    settings = LocalOpenAISettings.from_env()

    assert settings.model == "qwen-env"
    assert settings.base_url == "http://127.0.0.1:1234/v1"
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "test-only"
    assert settings.timeout_seconds == 42
    assert settings.disable_thinking is True


def test_local_adapter_requests_a_strict_schema_and_returns_a_neutral_result() -> None:
    transport = _RecordingTransport([_body(content=_assessment_json())])
    gateway = LocalOpenAIGateway(_settings(), transport=transport)

    result = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert result.output == passing_assessment()
    assert result.metadata.provider == "local_openai"
    assert result.metadata.model == "qwen-test"
    assert result.metadata.request_id == "chatcmpl-local-1"
    assert transport.urls == ["http://127.0.0.1:1234/v1/chat/completions"]
    schema = transport.requests[0]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["name"] == "LLMAssessment"
    assert schema["schema"] == LLMAssessment.model_json_schema()
    assert transport.requests[0]["messages"][0]["role"] == "system"
    assert "chat_template_kwargs" not in transport.requests[0]


def test_local_adapter_reads_structured_output_from_reasoning_content() -> None:
    """Reasoning builds place constrained JSON in reasoning_content, not content."""

    transport = _RecordingTransport([_body(content="", reasoning=_assessment_json())])
    gateway = LocalOpenAIGateway(_settings(), transport=transport)

    result = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert result.output == passing_assessment()


def test_local_adapter_strips_think_blocks_fences_and_trailing_prose() -> None:
    wrapped = f"<think>weighing the evidence</think>\n```json\n{_assessment_json()}\n```\nDone."
    transport = _RecordingTransport([_body(content=wrapped)])
    gateway = LocalOpenAIGateway(_settings(), transport=transport)

    result = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert result.output == passing_assessment()


def test_local_adapter_retries_with_the_validation_error_attached() -> None:
    transport = _RecordingTransport(
        [_body(content='{"verdict": "maybe"}'), _body(content=_assessment_json())]
    )
    gateway = LocalOpenAIGateway(_settings(), transport=transport)

    result = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert result.output == passing_assessment()
    repair = transport.requests[1]["messages"]
    assert repair[2]["role"] == "assistant"
    assert "did not satisfy the required schema" in repair[3]["content"]


def test_local_adapter_sends_the_thinking_switch_when_configured() -> None:
    transport = _RecordingTransport([_body(content=_assessment_json())])
    gateway = LocalOpenAIGateway(_settings(disable_thinking=True), transport=transport)

    gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert transport.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_local_adapter_fails_after_exhausting_retries() -> None:
    transport = _RecordingTransport([_body(content="not json at all")] * 3)
    gateway = LocalOpenAIGateway(_settings(max_retries=2), transport=transport)

    with pytest.raises(ModelInvocationError, match="invalid structured output"):
        gateway.invoke(
            task=ModelTask.GUARDRAIL_ASSESSMENT,
            prompt="Synthetic assessment input",
            response_model=LLMAssessment,
        )

    assert len(transport.requests) == 3


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"choices": []}, "no choices"),
        ({"choices": [{}]}, "no message"),
        ({"choices": [{"message": {"content": "  "}}]}, "empty message"),
    ],
)
def test_local_adapter_rejects_unusable_responses(body: dict[str, Any], message: str) -> None:
    gateway = LocalOpenAIGateway(_settings(), transport=_RecordingTransport([body]))

    with pytest.raises(ModelInvocationError, match=message):
        gateway.invoke(
            task=ModelTask.GUARDRAIL_ASSESSMENT,
            prompt="Synthetic assessment input",
            response_model=LLMAssessment,
        )


def test_urllib_transport_reports_a_dead_server_as_a_domain_error() -> None:
    gateway = LocalOpenAIGateway(
        LocalOpenAISettings(
            model="qwen-test",
            base_url="http://127.0.0.1:9/v1",
            timeout_seconds=1,
            api_key=SecretStr("test-only"),
        )
    )

    with pytest.raises(ModelInvocationError, match="failed"):
        gateway.invoke(
            task=ModelTask.GUARDRAIL_ASSESSMENT,
            prompt="Synthetic assessment input",
            response_model=LLMAssessment,
        )
