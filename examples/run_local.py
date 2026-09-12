"""Run the full workflow against a local OpenAI-compatible server.

Start LM Studio (or Ollama / vLLM) with a loaded model, then:

    LOCAL_LLM_MODEL=qwen/qwen3.8-27b uv run python examples/run_local.py
    LOCAL_LLM_MODEL=qwen/qwen3.8-27b uv run python examples/run_local.py \
        "data/sop_inputs/SOP Accounts - Payable.pdf"

With no path the built-in synthetic SOP is used. Reading a .pdf needs the
`documents` extra (`uv sync --extra documents`).

Both human gates are auto-approved here so the whole graph can be exercised in one
command. A real reviewer supplies those decisions; nothing about the gates is
bypassed inside the workflow itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from langgraph.types import Command

from sop_guardrail.application.workflow import DEFAULT_COMPILATION_BATCH_SIZE, build_workflow
from sop_guardrail.domain.models import ReviewDecision, ReviewGate, ReviewVerdict, SopDocument
from sop_guardrail.domain.segmentation import DEFAULT_MAX_SPAN_CHARS, segment_document
from sop_guardrail.infrastructure.documents import load_sop_document
from sop_guardrail.infrastructure.feedback import InMemoryFeedbackStore
from sop_guardrail.infrastructure.providers.local_openai import (
    LocalOpenAIGateway,
    LocalOpenAISettings,
)

SAMPLE_SOP = (
    "Any payment above 10,000 CHF must be approved by a compliance officer before the "
    "operator releases it. The approval record must identify the approving officer and "
    "the payment reference. Payments to a newly added beneficiary must be held for 24 "
    "hours before release."
)


def _approval(gate: ReviewGate) -> dict[str, object]:
    return ReviewDecision(
        gate=gate,
        verdict=ReviewVerdict.APPROVE,
        reviewer="local-operator",
        comment="Auto-approved by the local example runner.",
    ).model_dump(mode="json")


def _load_document(path: Path | None) -> SopDocument:
    if path is None:
        return SopDocument.from_text(
            document_id="local-demo-sop",
            source_name="payment-release-procedure.txt",
            text=SAMPLE_SOP,
        )
    return load_sop_document(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "document",
        nargs="?",
        type=Path,
        help="SOP file (.txt, .md, or .pdf); omit to use the built-in synthetic SOP",
    )
    parser.add_argument("--run-id", default="local-demo", help="run and checkpoint thread id")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_COMPILATION_BATCH_SIZE,
        help="policies per guardrail-compilation model call",
    )
    parser.add_argument(
        "--max-span-chars",
        type=int,
        default=DEFAULT_MAX_SPAN_CHARS,
        help="character cap before a long SOP section is split into more spans",
    )
    args = parser.parse_args(argv)

    settings = LocalOpenAISettings.from_env()
    print(f"provider: {settings.base_url} model={settings.model}", file=sys.stderr)

    document = _load_document(args.document)
    print(
        f"document: {document.document_id} ({len(document.text)} chars)"
        f" sha256={document.sha256[:12]}",
        file=sys.stderr,
    )
    print(
        f"evidence: {len(segment_document(document, max_span_chars=args.max_span_chars))} spans",
        file=sys.stderr,
    )

    graph = build_workflow(
        model_gateway=LocalOpenAIGateway(settings),
        feedback_store=InMemoryFeedbackStore(),
        compilation_batch_size=args.batch_size,
        max_span_chars=args.max_span_chars,
    )
    config = {"configurable": {"thread_id": args.run_id}}

    state = graph.invoke(
        {"run_id": args.run_id, "document": document.model_dump(mode="json")}, config
    )
    for gate in (ReviewGate.POLICY, ReviewGate.RELEASE):
        if "__interrupt__" not in state:
            break
        print(f"gate reached: {gate.value} -> approving", file=sys.stderr)
        state = graph.invoke(Command(resume=_approval(gate)), config)

    print(json.dumps({key: state[key] for key in state if key != "document"}, indent=2))
    if state.get("validation_errors"):
        print(f"validation errors: {state['validation_errors']}", file=sys.stderr)
        return 1
    return 0 if state.get("status") == "released" else 1


if __name__ == "__main__":
    os.environ.setdefault("LOCAL_LLM_MODEL", "qwen/qwen3.8-27b")
    raise SystemExit(main())
