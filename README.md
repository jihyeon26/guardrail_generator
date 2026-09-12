# SOP Guardrail Workbench

A provider-neutral, evidence-grounded agentic workflow that turns policy documents
into typed, reviewable guardrail specifications.

This repository is an independent public implementation built from public framework
documentation and synthetic examples. It contains no client source code, prompts,
schemas, data, screenshots, metrics, or Git history.

## Why this project exists

Long policy documents are difficult to translate into consistent controls. This
project treats that translation as a governed workflow rather than a single LLM call:

1. ingest an SOP and cut it into section-level evidence spans with exact offsets;
2. extract typed policy candidates that cite those spans;
3. pause for human review;
4. compile approved policies into typed guardrail rules;
5. run deterministic validation, including that every approved policy is enforced by
   at least one rule, then an advisory LLM review;
6. pause for final human approval;
7. publish an immutable, versioned guardrail release;
8. store review feedback for curated reuse in later runs.

LLM output never publishes a guardrail directly. Schema validation, evidence checks,
and explicit human approval form the release boundary.

## Technology choices

- Python 3.12+ with a `src/` package layout
- Pydantic v2 for boundary contracts and JSON Schema
- LangGraph for explicit workflow state and human interrupts
- Azure OpenAI / Microsoft Foundry through the GA OpenAI v1 endpoint
- A local OpenAI-compatible server (LM Studio, Ollama, vLLM) for offline development
- A provider protocol and deterministic fake provider for offline tests
- Optional `pypdf` text extraction for real SOP files
- `uv`, Ruff, mypy, pytest, and GitHub Actions for reproducible CI

Azure support is optional. The core package and test suite require no cloud account,
API key, or network access. The local adapter is part of the core package and adds no
dependencies beyond the standard library.

Python 3.12 is the minimum supported version, not a maximum. CI verifies Python 3.12,
3.13, and 3.14; keeping the local default at the minimum catches accidental use of
newer-only language features while allowing deployment on newer interpreters.

## Quick start

Install the project and development tools:

```powershell
uv sync --all-groups
```

Run the quality gates:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Install the optional Azure adapter:

```powershell
uv sync --extra azure --all-groups
```

Copy `.env.example` to `.env` locally and choose either Entra ID or an API key. Never
commit `.env`.

## Running against a local model

The local adapter needs no extra install. Start any OpenAI-compatible server, load a
model that supports JSON-schema-constrained decoding, and run the whole graph end to
end with both review gates auto-approved:

```powershell
$env:LOCAL_LLM_MODEL = "qwen/qwen3.8-27b"
uv run python examples/run_local.py
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible endpoint |
| `LOCAL_LLM_MODEL` | required | Model id as the server reports it |
| `LOCAL_LLM_TIMEOUT_SECONDS` | `600` | Per-call timeout |
| `LOCAL_LLM_DISABLE_THINKING` | unset | Send `enable_thinking: false` to reasoning models |
| `LOCAL_LLM_API_KEY` | unset | Only for servers that require a bearer token |

Guardrail compilation runs one model call per batch of policies (`--batch-size`,
default 5). A single call over a long policy list invites a model to answer with one
rule and stop. When a batch comes back incomplete it is re-asked with only the missing
policies (`--max-attempts`, default 3), so the list shrinks until it is the one-policy
request a small model does answer. Whatever is still missing after that fails the
coverage check rather than publishing a release that enforces a fraction of the SOP.

See [the local provider notes](docs/LOCAL_PROVIDER.md) for the behaviours that differ
from a hosted endpoint.

## Running against a real SOP file

Put the file in `data/sop_inputs/` and pass its path. `.txt` and `.md` need no extra;
`.pdf` needs the `documents` extra:

```powershell
uv sync --extra documents --all-groups
uv run python examples/run_local.py "data/sop_inputs/procedure.pdf" --run-id ap-v1
```

The loader normalizes the extracted text once, before evidence offsets are computed,
so a quoted span always matches the stored document. Scanned image-only PDFs are
rejected rather than producing empty evidence.

Ingest then splits the SOP on numbered headings into ordered, non-overlapping spans
(`--max-span-chars` caps a long section, default 1500). Each span records its
character range, its quote, and the quote's hash, and every policy must cite the spans
it was read from. Before a reviewer sees a policy, deterministic validation re-reads
each cited span out of the stored document and rejects any that no longer matches — a
citation is only worth as much as the span behind it.

The contents of `data/sop_inputs/` are git-ignored on purpose; see
[the folder README](data/sop_inputs/README.md) and [the clean-room
boundary](docs/CLEAN_ROOM.md).

## Repository map

```text
src/sop_guardrail/
  domain/          # Pydantic contracts, enums, and pure rules
  application/     # prompts, graph state, and LangGraph orchestration
  infrastructure/  # provider and repository adapters
tests/
  application/     # prompt, interrupt/resume, and graph routing tests
  domain/          # Pydantic contract and deterministic validation tests
  infrastructure/  # feedback storage and provider adapter tests
examples/
  run_local.py     # end-to-end run against a local OpenAI-compatible server
data/
  sop_inputs/      # untracked SOP files fed to the local runner
docs/
  PUBLIC_SPEC.md
  ARCHITECTURE.md
  PROVIDER_DECISION.md
  LOCAL_PROVIDER.md
  CLEAN_ROOM.md
```

See [the public specification](docs/PUBLIC_SPEC.md),
[architecture](docs/ARCHITECTURE.md), and
[provider decision](docs/PROVIDER_DECISION.md) before adding features.

## Project status

The first milestone is the executable workflow skeleton: typed contracts, two human
gates, an advisory LLM gate, curated feedback memory, Azure and local-model adapters
behind one provider boundary, text and PDF document loading, and credential-free CI
tests. A persistent database and a user interface are intentionally deferred until the
core state transitions are stable.

## License

MIT. See [LICENSE](LICENSE).
