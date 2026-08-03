# Provider Decision: Azure behind a portable gateway

## Decision

The first cloud adapter targets Azure OpenAI / Microsoft Foundry through the generally
available OpenAI v1 resource endpoint and LangChain's `ChatOpenAI`. The graph depends
only on a local `StructuredModelGateway` protocol.

## Why

- The OpenAI v1 endpoint uses deployment names as model aliases and avoids a dated
  `api-version` parameter.
- LangChain recommends `ChatOpenAI` for Azure's GA v1 API; the older
  `AzureChatOpenAI` path remains appropriate only for legacy versioned deployments.
- The same graph can use a deterministic fake in CI and another provider adapter later.
- Microsoft Entra ID supports keyless authentication and automatic token refresh.

## Authentication policy

- Deployed environments: Entra ID with least-privilege Azure RBAC.
- Local experiments: Entra ID preferred; an API key is permitted only through an
  untracked environment variable.
- CI: deterministic fake provider only; no live-provider secret is required.

## Explicitly avoided

- Azure AI Inference beta SDK, which Microsoft says retires on 2026-08-26;
- Azure-specific response objects or credentials in graph state and domain models;
- assuming every Foundry catalog model supports structured output or the Responses API.

## Sources

- [Microsoft Foundry model endpoints](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/endpoints)
- [LangChain Azure OpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/azure_chat_openai)
- [Microsoft structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)
