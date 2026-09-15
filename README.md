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
5. run deterministic validation — every approved policy is enforced by at least one
   rule, and every rule can actually refuse something — then an advisory LLM review;
6. pause for final human approval, with the assessment's findings in hand;
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

## Running a SOP through the workflow

`examples/run_local.py` runs the whole graph end to end, auto-approving both review
gates so one command exercises every node. A real reviewer supplies those decisions;
the gates themselves are not bypassed.

Start any OpenAI-compatible server (LM Studio, Ollama, vLLM) with a model that
supports JSON-schema-constrained decoding, then:

```powershell
$env:LOCAL_LLM_MODEL = "qwen/qwen3.8-27b"
uv sync --extra documents --all-groups        # .pdf input only
uv run python examples/run_local.py "data/sop_inputs/procedure.pdf" --run-id ap-v1
```

With no path the built-in synthetic SOP is used. `--provider demo` runs the graph in
seconds against the deterministic fake gateway, with no model server at all.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible endpoint |
| `LOCAL_LLM_MODEL` | required | Model id as the server reports it |
| `LOCAL_LLM_TIMEOUT_SECONDS` | `600` | Per-call timeout |
| `LOCAL_LLM_DISABLE_THINKING` | unset | Send `enable_thinking: false` to reasoning models |
| `LOCAL_LLM_API_KEY` | unset | Only for servers that require a bearer token |

The steps that call a model are batched, and `--batch-size`, `--max-attempts`,
`--assessment-batch-size`, `--concurrency`, and `--max-span-chars` tune them.
[The local provider notes](docs/LOCAL_PROVIDER.md) explain what each one is for and
what it costs, with measurements.

### Input files

Put SOP files in `data/sop_inputs/`. `.txt` and `.md` need no extra; `.pdf` needs the
`documents` extra. The loader normalizes extracted text once, before evidence offsets
are computed, so a quoted span always matches the stored document; scanned image-only
PDFs are rejected rather than producing empty evidence.

The folder's contents are git-ignored on purpose — see
[the folder README](data/sop_inputs/README.md) and
[the clean-room boundary](docs/CLEAN_ROOM.md).

### Exercising the feedback loop

A reviewer who sends a gate back can attach a reusable lesson. It is quarantined as
pending, and only a later run whose curator approved it will see it in a prompt.
`--feedback-file` keeps that quarantine across runs:

```powershell
uv run python examples/run_local.py <sop> --feedback-file data/runs/feedback.json `
  --revise-gate policy --lesson "A policy needs a checkable condition."
uv run python examples/run_local.py <sop> --feedback-file data/runs/feedback.json `
  --activate-feedback
```

The second command approves everything pending — standing in for a curator — and the
lesson reaches the extraction prompt of that run.

### Replaying one step

Every node's output is written to `data/runs/<run-id>/NN-<node>.json` as the run
streams. To work on one slow step without paying for the steps before it:

```powershell
uv run python examples/run_local.py "data/sop_inputs/procedure.pdf" `
  --run-id ap-v1 --resume-from validate_guardrails
```

These files are not a LangGraph checkpoint substitute; they serve development. They
stay readable, they can be edited between runs, and a single step can be re-run in
isolation. They do not survive a change to the domain contracts — a run recorded
before a required field was added will not replay.

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
  runs/            # untracked per-node output of each local run
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

The workflow runs end to end: a real multi-page SOP has been loaded, split into
evidence spans, extracted into policies, compiled into rules that cover every policy,
reviewed, and published as a versioned release, against a local model with no cloud
account.

Deferred on purpose: a persistent database, a user interface, and durable run history.
The curated feedback loop is implemented and tested but has not yet been exercised
across real runs.

## License

MIT. See [LICENSE](LICENSE).
