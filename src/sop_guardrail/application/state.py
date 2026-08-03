"""Serialization-friendly LangGraph state."""

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    run_id: str
    document: dict[str, Any]
    evidence: list[dict[str, Any]]
    policies: list[dict[str, Any]]
    guardrails: list[dict[str, Any]]
    llm_assessment: dict[str, Any]
    model_runs: list[dict[str, Any]]
    policy_review: dict[str, Any]
    release_review: dict[str, Any]
    validation_errors: list[str]
    feedback_ids: list[str]
    release: dict[str, Any]
    status: str
