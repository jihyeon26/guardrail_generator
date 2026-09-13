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


def _document_header(document: BaseModel) -> str:
    """Identify the document without repeating its text; the spans already carry it."""

    payload = {
        key: value for key, value in document.model_dump(mode="json").items() if key != "text"
    }
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
        "Extract only normative obligations. A policy requires a named actor to do "
        "something, or forbids it, in a way a reviewer could check.\n"
        "Do not extract, and do not rewrite into a policy:\n"
        "- permissions and options ('may', 'can', 'is allowed to', 'depending on preference');\n"
        "- efficiency or convenience guidance ('to streamline', 'to take advantage of');\n"
        "- definitions, scope statements, and descriptions of who a role is;\n"
        "- a restatement of an obligation already extracted.\n"
        "Set modality to 'required' or 'prohibited'. If neither fits the text, it is not a "
        "policy; leave it out. Prefer fewer checkable policies over covering every sentence.\n"
        "Do not infer missing obligations. The evidence spans below are the SOP, split into "
        "sections. Every policy must cite the evidence_id values of the spans it was read "
        "from, and no others.\n\n"
        f"DOCUMENT\n{_document_header(document)}\n\n"
        f"EVIDENCE ({len(evidence)} spans)\n{_json(evidence)}\n\n"
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
        "coverage gaps, ambiguity, and unverifiable conditions. Abstain if evidence is thin. "
        "This assessment is advisory and cannot approve a release.\n"
        "The findings are the point of this review: report every material problem you find, "
        "up to five, each one sentence naming the rule it concerns. A review that finds nothing "
        "worth reporting should say so in the summary.\n"
        "Spend words on findings, not around them. Keep the summary to one sentence. Give "
        "uncertainty only when it would change the verdict. Do not restate the rules, quote the "
        "evidence back, or describe the task you were given.\n\n"
        f"POLICIES\n{_json(policies)}\n\nGUARDRAILS\n{_json(guardrails)}\n\n"
        f"EVIDENCE\n{_json(evidence)}\n\n"
        f"APPROVED_FEEDBACK\n{_feedback_section(feedback)}"
    )
