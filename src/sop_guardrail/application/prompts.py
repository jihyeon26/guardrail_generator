"""Original public prompts assembled from typed domain artifacts."""

import json

from pydantic import BaseModel

from sop_guardrail.domain.models import FeedbackCard


def _json(value: BaseModel | tuple[BaseModel, ...]) -> str:
    if isinstance(value, tuple):
        payload: object = [item.model_dump(mode="json") for item in value]
    else:
        payload = value.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _feedback_section(feedback: tuple[FeedbackCard, ...]) -> str:
    if not feedback:
        return "No approved lessons apply to this task."
    lessons = "\n".join(f"- [{item.feedback_id}] {item.lesson}" for item in feedback)
    return f"Apply only these human-approved lessons when relevant:\n{lessons}"


def policy_extraction_prompt(
    *, document: BaseModel, evidence: tuple[BaseModel, ...], feedback: tuple[FeedbackCard, ...]
) -> str:
    return (
        "Extract explicit normative policies from the SOP. Do not infer missing obligations. "
        "Every policy must cite one or more evidence_id values from the supplied evidence.\n\n"
        f"DOCUMENT\n{_json(document)}\n\nEVIDENCE\n{_json(evidence)}\n\n"
        f"APPROVED_FEEDBACK\n{_feedback_section(feedback)}"
    )


def guardrail_compilation_prompt(
    *, policies: tuple[BaseModel, ...], feedback: tuple[FeedbackCard, ...]
) -> str:
    return (
        "Compile each approved policy into a testable rule. Preserve policy_id and evidence_refs. "
        "Use escalation when the policy cannot be enforced deterministically, and provide at least "
        "one test case per rule.\n"
        "Return one rule for every policy listed below and no rule for any other policy. "
        "Policies arrive in batches, so derive each rule_id from the policy_id it enforces to "
        "keep rule ids unique across the whole run.\n\n"
        f"POLICIES ({len(policies)} to compile)\n{_json(policies)}\n\n"
        f"APPROVED_FEEDBACK\n{_feedback_section(feedback)}"
    )


def assessment_prompt(
    *,
    policies: tuple[BaseModel, ...],
    guardrails: tuple[BaseModel, ...],
    evidence: tuple[BaseModel, ...],
    feedback: tuple[FeedbackCard, ...],
) -> str:
    return (
        "Review proposed guardrails against supplied evidence. Identify unsupported claims, "
        "Find coverage gaps, ambiguity, and unverifiable conditions. Abstain if evidence is thin. "
        "This assessment is advisory and cannot approve a release.\n\n"
        f"POLICIES\n{_json(policies)}\n\nGUARDRAILS\n{_json(guardrails)}\n\n"
        f"EVIDENCE\n{_json(evidence)}\n\n"
        f"APPROVED_FEEDBACK\n{_feedback_section(feedback)}"
    )
