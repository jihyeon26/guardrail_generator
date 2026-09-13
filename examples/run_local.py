"""Run the full workflow against a local OpenAI-compatible server.

Start LM Studio (or Ollama / vLLM) with a loaded model, then:

    LOCAL_LLM_MODEL=qwen/qwen3.8-27b uv run python examples/run_local.py
    LOCAL_LLM_MODEL=qwen/qwen3.8-27b uv run python examples/run_local.py \
        "data/sop_inputs/SOP Accounts - Payable.pdf"

With no path the built-in synthetic SOP is used. Reading a .pdf needs the
`documents` extra (`uv sync --extra documents`).

Every node's output is written to ``data/runs/<run-id>/NN-<node>.json``. To work on
one slow step without paying for the steps before it:

    uv run python examples/run_local.py <sop> --run-id ap-v3 --resume-from validate_guardrails

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
from typing import Any, cast

from langgraph.types import Command

from sop_guardrail.application.workflow import (
    DEFAULT_COMPILATION_BATCH_SIZE,
    DEFAULT_COMPILATION_MAX_ATTEMPTS,
    build_workflow,
)
from sop_guardrail.domain.models import ReviewDecision, ReviewGate, ReviewVerdict, SopDocument
from sop_guardrail.domain.segmentation import DEFAULT_MAX_SPAN_CHARS, segment_document
from sop_guardrail.infrastructure.artifacts import RunArtifactStore
from sop_guardrail.infrastructure.documents import load_sop_document
from sop_guardrail.infrastructure.feedback import InMemoryFeedbackStore
from sop_guardrail.infrastructure.providers.demo import DemoModelGateway
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


class _Recorder:
    """Stream node updates to disk, then read the settled state back."""

    def __init__(self, artifacts: RunArtifactStore) -> None:
        self.artifacts = artifacts
        self.index = 0

    def run(self, graph: Any, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node.startswith("__") or not isinstance(update, dict):
                    continue
                path = self.artifacts.write(index=self.index, node=node, update=update)
                print(f"  step {self.index:02d} {node} -> {path}", file=sys.stderr)
                self.index += 1
        return cast(dict[str, Any], graph.get_state(config).values)


def _pending_gate(graph: Any, config: dict[str, Any]) -> ReviewGate | None:
    """Which gate the graph is waiting on, read from the interrupt itself.

    A resumed run can arrive at the release gate without ever touching the policy
    gate, so the gate is taken from the pending request rather than assumed.
    """

    for task in graph.get_state(config).tasks:
        for pending in getattr(task, "interrupts", ()):
            value = getattr(pending, "value", None)
            if isinstance(value, dict) and "gate" in value:
                return ReviewGate(value["gate"])
    return None


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
        "--artifacts-dir",
        type=Path,
        default=None,
        help="where per-node output is written (default: data/runs/<run-id>)",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "demo"),
        default="local",
        help="'demo' uses the deterministic fake gateway to exercise the graph offline",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="replay recorded steps through this node, then continue after it",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_COMPILATION_BATCH_SIZE,
        help="policies per guardrail-compilation model call",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_COMPILATION_MAX_ATTEMPTS,
        help="attempts per batch before the missing policies are reported",
    )
    parser.add_argument(
        "--max-span-chars",
        type=int,
        default=DEFAULT_MAX_SPAN_CHARS,
        help="character cap before a long SOP section is split into more spans",
    )
    args = parser.parse_args(argv)

    if args.provider == "demo":
        gateway: Any = DemoModelGateway()
        print("provider: deterministic demo gateway", file=sys.stderr)
    else:
        settings = LocalOpenAISettings.from_env()
        gateway = LocalOpenAIGateway(settings)
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
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        compilation_batch_size=args.batch_size,
        compilation_max_attempts=args.max_attempts,
        max_span_chars=args.max_span_chars,
    )
    config: dict[str, Any] = {"configurable": {"thread_id": args.run_id}}
    artifacts = RunArtifactStore(args.artifacts_dir or Path("data/runs") / args.run_id)
    recorder = _Recorder(artifacts)

    if args.resume_from is None:
        start: Any = {"run_id": args.run_id, "document": document.model_dump(mode="json")}
    else:
        replayed, as_node = artifacts.replay_through(args.resume_from)
        replayed.setdefault("run_id", args.run_id)
        replayed.setdefault("document", document.model_dump(mode="json"))
        graph.update_state(config, replayed, as_node=as_node)
        recorder.index = len(artifacts.steps())
        print(f"resumed after node: {as_node}", file=sys.stderr)
        start = None

    state = recorder.run(graph, start, config)
    for _ in range(len(ReviewGate)):
        gate = _pending_gate(graph, config)
        if gate is None:
            break
        print(f"gate reached: {gate.value} -> approving", file=sys.stderr)
        state = recorder.run(graph, Command(resume=_approval(gate)), config)

    print(json.dumps({key: state[key] for key in state if key != "document"}, indent=2))
    if state.get("validation_errors"):
        print(f"validation errors: {state['validation_errors']}", file=sys.stderr)
        return 1
    return 0 if state.get("status") == "released" else 1


if __name__ == "__main__":
    os.environ.setdefault("LOCAL_LLM_MODEL", "qwen/qwen3.8-27b")
    raise SystemExit(main())
