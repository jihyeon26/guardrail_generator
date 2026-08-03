import sop_guardrail


def test_public_package_exports_are_available() -> None:
    assert sop_guardrail.__version__ == "0.1.0"
    assert callable(sop_guardrail.build_workflow)
