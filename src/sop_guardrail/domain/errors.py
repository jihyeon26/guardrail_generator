"""Domain-specific failures surfaced by adapters and validators."""


class GuardrailWorkbenchError(Exception):
    """Base error for expected workflow failures."""


class ModelInvocationError(GuardrailWorkbenchError):
    """A provider call failed or returned unusable structured output."""


class ReferenceValidationError(GuardrailWorkbenchError):
    """A typed artifact references an unknown or duplicate identifier."""
