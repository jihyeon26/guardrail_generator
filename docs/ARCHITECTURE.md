# Architecture

## Trust boundaries

The workflow separates five concerns:

1. **Evidence**: immutable source identity and stable text spans.
2. **Specification**: typed policy candidates and guardrail rules.
3. **Decision**: explicit human review records and advisory LLM assessments.
4. **Release**: immutable, versioned guardrail artifacts.
5. **Learning**: quarantined feedback promoted only after approval.

An LLM may propose structured data but cannot write a release or activate feedback.

## Workflow

```text
START
  -> ingest
  -> extract policies
  -> deterministic policy validation
  -> human policy gate
       revise/reject -> record pending feedback -> END
       approve       -> compile guardrails
  -> deterministic rule validation
  -> advisory LLM assessment
  -> final human gate
       revise/reject -> record pending feedback -> END
       approve       -> publish immutable release -> END
```

LangGraph state contains JSON-serializable domain snapshots and identifiers. Provider
clients, credentials, and repository connections are injected dependencies and never
become graph state.

## Package boundaries

- `domain`: Pydantic contracts, enums, errors, ports, and pure validation.
- `application`: prompt builders, graph state, nodes, and routing.
- `infrastructure`: Azure, fake-model, feedback-store, and checkpointer adapters.

The domain does not import LangGraph, LangChain, Azure, or storage implementations.

## Persistence roadmap

The initial implementation uses LangGraph's in-memory checkpointer and an in-memory
feedback store for deterministic tests. A later milestone will add SQLite for a local
single-user demo and Postgres for durable multi-user deployment. Workflow checkpoints
support pause/resume; they do not replace application-level idempotency or an audit log.

## Testing strategy

- unit tests for Pydantic invariants, evidence-reference validation, and prompt assembly;
- graph tests for each routing branch and both interrupt/resume boundaries;
- provider contract tests using deterministic structured outputs;
- no-network CI by default;
- optional live Azure smoke tests only in a separately protected workflow.
