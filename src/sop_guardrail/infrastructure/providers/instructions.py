"""Task instructions shared by every structured model adapter."""

from __future__ import annotations

from sop_guardrail.domain.models import ModelTask

SYSTEM_INSTRUCTIONS: dict[ModelTask, str] = {
    ModelTask.POLICY_EXTRACTION: (
        "Return only schema-conforming policies explicitly supported by supplied evidence."
    ),
    ModelTask.GUARDRAIL_COMPILATION: (
        "Return only schema-conforming guardrails derived from approved policies."
    ),
    ModelTask.GUARDRAIL_ASSESSMENT: (
        "Act as an advisory evidence reviewer. Report uncertainty and never claim final approval."
    ),
}
