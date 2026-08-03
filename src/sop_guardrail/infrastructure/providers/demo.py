"""Credential-free providers for demonstrations and deterministic tests."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import TypeVar

from sop_guardrail.domain.models import (
    AssessmentVerdict,
    ContractModel,
    GuardrailCompilation,
    GuardrailDecision,
    GuardrailRule,
    LLMAssessment,
    ModelMetadata,
    ModelResult,
    ModelTask,
    PolicyCandidate,
    PolicyExtraction,
    RuleTestCase,
    Severity,
)

OutputT = TypeVar("OutputT", bound=ContractModel)


class ScriptedModelGateway:
    """Return queued typed responses and retain prompts for assertions."""

    def __init__(self, responses: Mapping[ModelTask, Sequence[ContractModel]]) -> None:
        self._responses = {task: deque(items) for task, items in responses.items()}
        self.prompts: dict[ModelTask, list[str]] = defaultdict(list)

    def invoke(
        self,
        *,
        task: ModelTask,
        prompt: str,
        response_model: type[OutputT],
    ) -> ModelResult[OutputT]:
        self.prompts[task].append(prompt)
        if task not in self._responses or not self._responses[task]:
            raise AssertionError(f"no scripted response remains for task {task}")
        raw = self._responses[task].popleft()
        output = response_model.model_validate(raw.model_dump(mode="json"))
        return ModelResult[OutputT](
            output=output,
            metadata=ModelMetadata(provider="scripted", model="fixture-v1"),
        )


class DemoModelGateway:
    """Small deterministic model substitute; it is not a semantic SOP parser."""

    def __init__(self) -> None:
        self.prompts: dict[ModelTask, list[str]] = defaultdict(list)

    @staticmethod
    def _first_id(prompt: str, field: str) -> str:
        matches: list[str] = re.findall(rf'"{field}":\s*"([^"]+)"', prompt)
        if not matches:
            raise ValueError(f"prompt does not contain {field}")
        return str(matches[0])

    @staticmethod
    def _first_list_item(prompt: str, field: str) -> str:
        match = re.search(rf'"{field}":\s*\[\s*"([^"]+)"', prompt)
        if match is None:
            raise ValueError(f"prompt does not contain a value in {field}")
        return match.group(1)

    def invoke(
        self,
        *,
        task: ModelTask,
        prompt: str,
        response_model: type[OutputT],
    ) -> ModelResult[OutputT]:
        self.prompts[task].append(prompt)

        if task is ModelTask.POLICY_EXTRACTION:
            evidence_id = self._first_id(prompt, "evidence_id")
            value: ContractModel = PolicyExtraction(
                policies=(
                    PolicyCandidate(
                        policy_id="policy-demo-1",
                        title="Manual review requirement",
                        statement="The described operation requires an authorized review.",
                        actor="authorized reviewer",
                        action="review the operation",
                        condition="before completion",
                        evidence_refs=(evidence_id,),
                        confidence=0.8,
                    ),
                )
            )
        elif task is ModelTask.GUARDRAIL_COMPILATION:
            policy_id = self._first_id(prompt, "policy_id")
            evidence_id = self._first_list_item(prompt, "evidence_refs")
            value = GuardrailCompilation(
                rules=(
                    GuardrailRule(
                        rule_id="rule-demo-1",
                        policy_id=policy_id,
                        decision=GuardrailDecision.ESCALATE,
                        condition="an operation reaches the review checkpoint",
                        rationale="The policy requires an authorized human review.",
                        severity=Severity.HIGH,
                        evidence_refs=(evidence_id,),
                        test_cases=(
                            RuleTestCase(
                                name="review is required",
                                input_summary="operation reaches the checkpoint without approval",
                                expected_decision=GuardrailDecision.ESCALATE,
                            ),
                        ),
                    ),
                )
            )
        else:
            value = LLMAssessment(
                verdict=AssessmentVerdict.PASS,
                summary="The rule remains linked to the supplied evidence.",
                uncertainty="A domain reviewer must confirm the intended control.",
            )

        output = response_model.model_validate(value.model_dump(mode="json"))
        return ModelResult[OutputT](
            output=output,
            metadata=ModelMetadata(provider="demo", model="deterministic-v1"),
        )
