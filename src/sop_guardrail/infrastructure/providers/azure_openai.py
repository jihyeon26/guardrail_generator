"""Optional Azure OpenAI / Microsoft Foundry OpenAI v1 adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from enum import StrEnum
from time import perf_counter
from typing import Any, TypeVar, cast

from pydantic import Field, SecretStr, model_validator

from sop_guardrail.domain.errors import ModelInvocationError
from sop_guardrail.domain.models import (
    ContractModel,
    ModelMetadata,
    ModelResult,
    ModelTask,
)
from sop_guardrail.infrastructure.providers.instructions import SYSTEM_INSTRUCTIONS

OutputT = TypeVar("OutputT", bound=ContractModel)


class AzureAuthMode(StrEnum):
    ENTRA = "entra"
    API_KEY = "api_key"


class AzureOpenAIV1Settings(ContractModel):
    base_url: str = Field(min_length=1)
    deployment: str = Field(min_length=1)
    auth_mode: AzureAuthMode = AzureAuthMode.ENTRA
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @model_validator(mode="after")
    def validate_endpoint_and_auth(self) -> AzureOpenAIV1Settings:
        if not self.base_url.endswith("/openai/v1/"):
            raise ValueError("base_url must end with /openai/v1/")
        if self.auth_mode is AzureAuthMode.API_KEY and self.api_key is None:
            raise ValueError("api_key is required when auth_mode=api_key")
        return self

    @classmethod
    def from_env(cls) -> AzureOpenAIV1Settings:
        raw_key = os.getenv("AZURE_OPENAI_API_KEY")
        return cls(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            auth_mode=AzureAuthMode(os.getenv("AZURE_OPENAI_AUTH_MODE", "entra")),
            api_key=SecretStr(raw_key) if raw_key else None,
        )


class AzureOpenAIV1Gateway:
    """Translate provider responses into local Pydantic contracts."""

    def __init__(self, settings: AzureOpenAIV1Settings, *, chat_model: Any | None = None) -> None:
        self.settings = settings
        self._chat_model = chat_model if chat_model is not None else self._build_chat_model()

    def _build_chat_model(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("install the project with the 'azure' extra") from exc

        credential: SecretStr | Callable[[], str]
        if self.settings.auth_mode is AzureAuthMode.ENTRA:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise RuntimeError("install the project with the 'azure' extra") from exc
            credential = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
        else:
            if self.settings.api_key is None:  # defensive; settings validation already rejects it
                raise ValueError("missing API key")
            credential = self.settings.api_key

        return ChatOpenAI(
            model=self.settings.deployment,
            base_url=self.settings.base_url,
            api_key=credential,
            timeout=self.settings.timeout_seconds,
            max_retries=self.settings.max_retries,
        )

    def invoke(
        self,
        *,
        task: ModelTask,
        prompt: str,
        response_model: type[OutputT],
    ) -> ModelResult[OutputT]:
        structured = self._chat_model.with_structured_output(response_model, include_raw=True)
        started = perf_counter()
        try:
            payload = cast(
                dict[str, Any],
                structured.invoke(
                    [
                        ("system", SYSTEM_INSTRUCTIONS[task]),
                        ("human", prompt),
                    ]
                ),
            )
        except Exception as exc:  # providers expose several transport-specific failures
            raise ModelInvocationError(f"Azure model invocation failed for {task}") from exc

        if payload.get("parsing_error") is not None:
            raise ModelInvocationError(f"Azure returned invalid structured output for {task}")
        parsed = payload.get("parsed")
        if parsed is None:
            raise ModelInvocationError(f"Azure returned no structured output for {task}")
        output = response_model.model_validate(parsed)

        raw = payload.get("raw")
        metadata = getattr(raw, "response_metadata", {}) if raw is not None else {}
        request_id = metadata.get("request_id") or metadata.get("x-request-id")
        return ModelResult[OutputT](
            output=output,
            metadata=ModelMetadata(
                provider="azure_openai_v1",
                model=self.settings.deployment,
                request_id=str(request_id) if request_id else None,
                latency_ms=round((perf_counter() - started) * 1000),
            ),
        )
