# Public Product Specification

## Objective

Convert a synthetic or publicly licensed standard operating procedure into an
evidence-linked, typed, testable guardrail release through an explicit workflow with
deterministic, LLM, and human review boundaries.

## Non-goals

- reproducing any private implementation or client workflow;
- treating schema-valid output as factually correct;
- letting an LLM publish or enforce a guardrail autonomously;
- using private documents, evaluation fixtures, prompts, schemas, or performance data;
- providing legal or regulatory advice.

## Functional requirements

1. Every policy candidate must cite one or more stable evidence-span identifiers.
2. Every guardrail rule must cite approved policies and evidence spans.
3. Unknown evidence identifiers, duplicate identifiers, and invalid transitions fail
   deterministic validation.
4. Every approved policy must be enforced by at least one guardrail rule; a compilation
   that silently drops policies fails deterministic validation.
5. Policy candidates require an explicit human approve, revise, or reject decision.
6. Guardrail releases require an explicit final human approval after LLM assessment.
7. LLM assessment is advisory and must report uncertainty and cited evidence.
8. Review feedback starts as pending and cannot affect later runs until approved.
9. Each model call and released artifact carries provider-neutral provenance metadata.
10. Tests run without cloud credentials or network access.

## Initial acceptance criteria

- a happy-path graph can pause and resume at both human gates;
- rejected or revision-requested work does not publish a release;
- malformed model output or unknown evidence references cannot reach a human gate;
- a guardrail set that leaves an approved policy unenforced cannot reach a human gate;
- approved feedback is included in a later task prompt while pending feedback is not;
- the Azure adapter is optional and never imported during the default test run;
- CI checks formatting, lint, types, tests, coverage, and package build.
