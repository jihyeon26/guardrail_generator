# Local Provider: an OpenAI-compatible server behind the same gateway

## Decision

Development and experimentation run against a locally hosted OpenAI-compatible
server — LM Studio, Ollama, or vLLM — through `LocalOpenAIGateway`. It implements the
same `StructuredModelGateway` protocol as the Azure adapter, so the graph, prompts,
contracts, validators, and human gates are untouched by the choice of provider.

## Why

- Iterating on prompts and contracts needs no cloud account, no credential, and no
  per-token cost.
- Synthetic SOP text never leaves the machine, which keeps the clean-room boundary
  trivially verifiable.
- Constrained decoding on a local server enforces the same Pydantic JSON Schema the
  hosted provider is asked to honour, so contract drift shows up in development.
- The adapter uses only the standard library, so `LocalOpenAISettings` and
  `LocalOpenAIGateway` stay importable in the default install and the default CI run.

## How structured output is obtained

The adapter sends the Pydantic JSON Schema in `response_format.json_schema` with
`strict: true`. Pydantic `$defs`/`$ref` schemas, `anyOf` optionals, and array
constraints are accepted by LM Studio's constrained decoder as-is; no schema flattening
is required.

## Local behaviours the adapter absorbs

These differ from a hosted OpenAI-compatible endpoint and are handled in the adapter,
not in the workflow:

- **Constrained JSON arrives in `reasoning_content`.** A reasoning model under a JSON
  grammar cannot emit its closing think delimiter, so the server classifies the whole
  completion as reasoning and leaves `content` empty. The adapter reads `content`
  first and falls back to `reasoning_content`.
- **Wrapped payloads.** `<think>` blocks, Markdown code fences, and trailing prose are
  stripped, and the first complete JSON object is decoded.
- **Schema-valid but contract-invalid output.** Grammar conformance does not imply
  the field constraints hold (`min_length`, enum members, evidence references). A
  failed `model_validate` is retried up to `max_retries` times with the validation
  error appended to the conversation, then raised as `ModelInvocationError`.
- **Latency.** A single assessment call on a 27B model can exceed five minutes, so the
  default timeout is 600 seconds rather than the hosted adapter's 60.
- **Thinking cost.** `LOCAL_LLM_DISABLE_THINKING=true` sends
  `chat_template_kwargs: {"enable_thinking": false}`, which large reasoning models
  honour and other models ignore.

## Explicitly avoided

- A second HTTP dependency: the transport is `urllib.request` behind a
  `ChatCompletionTransport` protocol that tests substitute.
- Provider branching inside workflow nodes; the gateway choice is made once at
  composition time.
- Live local-model calls in CI, which would make the test suite depend on a running
  server and a downloaded model.

## Verification

`examples/run_local.py` runs ingest → extraction → validation → policy gate →
compilation → validation → advisory assessment → release gate → publish against the
configured server and exits non-zero unless the run reaches `status: released`.
