"""Evidence-grounded SOP-to-guardrail workflow."""

from sop_guardrail.application.workflow import build_workflow
from sop_guardrail.domain.models import SopDocument

__all__ = ["SopDocument", "build_workflow"]
__version__ = "0.1.0"
