from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from sop_guardrail.domain.errors import ModelInvocationError
from sop_guardrail.domain.models import LLMAssessment, ModelTask
from sop_guardrail.infrastructure.providers.azure_openai import (
    AzureAuthMode,
    AzureOpenAIV1Gateway,
    AzureOpenAIV1Settings,
)
from tests.helpers import passing_assessment


class _RawMessage:
    response_metadata = {"request_id": "request-123"}


class _StructuredModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert messages[0][0] == "system"
        return self.payload


class _ChatModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def with_structured_output(
        self, response_model: type[LLMAssessment], *, include_raw: bool
    ) -> _StructuredModel:
        assert response_model is LLMAssessment
        assert include_raw is True
        return _StructuredModel(self.payload)


def _settings() -> AzureOpenAIV1Settings:
    return AzureOpenAIV1Settings(
        base_url="https://example.openai.azure.com/openai/v1/",
        deployment="portfolio-model",
        auth_mode=AzureAuthMode.API_KEY,
        api_key=SecretStr("test-only"),
    )


def test_settings_require_v1_endpoint_suffix() -> None:
    with pytest.raises(ValidationError, match="/openai/v1/"):
        AzureOpenAIV1Settings(
            base_url="https://example.openai.azure.com/",
            deployment="portfolio-model",
        )


def test_injected_azure_model_returns_provider_neutral_result() -> None:
    gateway = AzureOpenAIV1Gateway(
        _settings(),
        chat_model=_ChatModel(
            {
                "parsed": passing_assessment(),
                "raw": _RawMessage(),
                "parsing_error": None,
            }
        ),
    )

    result = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt="Synthetic assessment input",
        response_model=LLMAssessment,
    )

    assert result.output == passing_assessment()
    assert result.metadata.provider == "azure_openai_v1"
    assert result.metadata.request_id == "request-123"


def test_azure_adapter_rejects_parsing_error() -> None:
    gateway = AzureOpenAIV1Gateway(
        _settings(),
        chat_model=_ChatModel(
            {"parsed": None, "raw": _RawMessage(), "parsing_error": ValueError("bad")}
        ),
    )

    with pytest.raises(ModelInvocationError, match="invalid structured output"):
        gateway.invoke(
            task=ModelTask.GUARDRAIL_ASSESSMENT,
            prompt="Synthetic assessment input",
            response_model=LLMAssessment,
        )
