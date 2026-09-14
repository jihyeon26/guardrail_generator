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
- **Latency.** A single call on a 27B model can exceed five minutes, so the default
  timeout is 600 seconds rather than the hosted adapter's 60. Raise it further when
  running batches concurrently: a request's clock includes time spent queued.
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

## Working within a local model's budget

A local 27B model generates at single-digit tokens per second, so every step that
calls it is shaped around that. These are not provider quirks the adapter can hide;
they are workflow decisions, tuned by the flags on `examples/run_local.py`.

**Compilation is batched, and an incomplete batch is re-asked.** Asked to compile a
long policy list in one call, the model returns one rule and stops — it satisfies the
schema, which only requires a non-empty list. Batching (`--batch-size`, default 5)
shrinks the request; when a batch still comes back short, it is re-asked with only the
policies it missed (`--max-attempts`, default 3) until the list is the single-policy
request a small model does answer. Whatever is still missing fails the coverage check
rather than publishing a release that enforces a fraction of the SOP.

**Assessment is batched by rule, carrying only the context those rules cite**
(`--assessment-batch-size`, default 5). Reviewing every rule in one call built an
11,000-token prompt the model could not finish; the largest batched prompt is about
2,300 tokens, and that ceiling does not rise with the length of the SOP. Batch
verdicts merge worst-first, so a clean batch cannot soften one that rejected.

**The assessment's output budget is spent on findings.** Assessment cost is dominated
by generation, not prompt size. Told only to be brief, the model cut the findings —
the part worth reading — and kept its prose: 95 output tokens, zero findings. Told
that findings are the point and that the summary is what should be short, the same
batch produced five specific findings in 498 tokens, against 1,494 tokens before any
instruction.

**Compilation asks for the violation, not the compliant case.** Left to itself the
model restates the policy as the situation that is fine and allows it — a rule that
cites its policy and evidence, carries a test case, and can never fire. Nineteen of
thirty-two rules in one run were this shape. The prompt now asks for the condition
that must be stopped, and deterministic validation rejects any rule that decides
`allow` or whose test cases never expect a refusal.

**Extraction asks for obligations, not sentences.** `PolicyCandidate.modality` forces
each policy to be `required` or `prohibited`. A permission ("payments can be made via
EFT") fits neither, which is what keeps it out; a rule cannot be written for it, and
before the field existed such entries failed the coverage check three retries later.

**Concurrency helps unevenly.** Batches within a step are independent and run
concurrently (`--concurrency`, default 4 in the runner; the workflow itself defaults
to sequential so a hosted provider's rate limit is never hit by surprise). Results are
collected in batch order either way. The gain depends on what the step spends time on:
about 1.85x on compilation-shaped calls, which are dominated by prompt processing, and
little to nothing on assessment calls, which are dominated by generation.

### Measurements

One machine, LM Studio, `qwen/qwen3.8-27b`, thinking disabled, a 5,798-character SOP
split into 12 spans. Indicative, not a benchmark.

| | |
| --- | --- |
| Full run, 32 policies → 32 rules → release | 2,560s wall, 7,583s of model-call time |
| Extraction (one call, not parallelizable) | 658s |
| Compilation (9 calls including retries) | 2,334s |
| Advisory assessment (7 calls) | 4,591s |
| Concurrency on compilation-shaped calls | 1.85x at four in flight, no gain at eight |
| Re-sending the same assessment prompts | 2.8x faster — LM Studio caches prompt prefixes |

Assessment is the largest cost, and reducing the work is the lever that matters:
fewer rules, or shorter findings, not a different concurrency setting. Two caveats on
these numbers. The concurrency comparison for assessment was run against a warm prompt
cache and is not clean enough to tune on. And `--resume-from` reuses prompts a previous
run already sent, so a replayed step is cheaper than the same step run cold.

## Verification

`examples/run_local.py` runs ingest → extraction → validation → policy gate →
compilation → validation → advisory assessment → release gate → publish against the
configured server and exits non-zero unless the run reaches `status: released`.
