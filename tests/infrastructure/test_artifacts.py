import json
from pathlib import Path

import pytest

from sop_guardrail.infrastructure.artifacts import RunArtifactStore


def _store(tmp_path: Path) -> RunArtifactStore:
    return RunArtifactStore(tmp_path / "run-1")


def test_a_step_is_written_under_an_ordered_readable_name(tmp_path: Path) -> None:
    store = _store(tmp_path)

    path = store.write(index=3, node="compile_guardrails", update={"status": "guardrails_compiled"})

    assert path.name == "03-compile_guardrails.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "guardrails_compiled"}


def test_steps_are_returned_in_index_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(index=1, node="extract_policies", update={"policies": []})
    store.write(index=0, node="ingest", update={"evidence": []})
    store.write(index=10, node="publish", update={"status": "released"})

    assert [step.node for step in store.steps()] == ["ingest", "extract_policies", "publish"]


def test_unrelated_files_are_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(index=0, node="ingest", update={"evidence": []})
    (store.directory / "notes.md").write_text("scratch", encoding="utf-8")
    (store.directory / "9-bad-name.json").write_text("{}", encoding="utf-8")

    assert [step.node for step in store.steps()] == ["ingest"]


def test_a_directory_with_no_run_yet_has_no_steps(tmp_path: Path) -> None:
    assert _store(tmp_path).steps() == ()


def test_replay_merges_every_step_up_to_the_named_node(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(index=0, node="ingest", update={"evidence": ["span"], "status": "ingested"})
    store.write(index=1, node="extract_policies", update={"policies": ["p"], "status": "extracted"})
    store.write(index=2, node="validate_policies", update={"status": "policies_validated"})
    store.write(index=3, node="policy_review", update={"status": "policy_approve"})

    state, as_node = store.replay_through("validate_policies")

    assert as_node == "validate_policies"
    assert state == {
        "evidence": ["span"],
        "policies": ["p"],
        "status": "policies_validated",
    }


def test_replay_uses_the_last_run_of_a_repeated_node(tmp_path: Path) -> None:
    """A resumed run appends a second pass of the same node; the later one wins."""

    store = _store(tmp_path)
    store.write(index=0, node="llm_assessment", update={"llm_assessment": {"verdict": "pass"}})
    store.write(index=1, node="release_review", update={"status": "release_approve"})
    store.write(index=2, node="llm_assessment", update={"llm_assessment": {"verdict": "revise"}})

    state, _ = store.replay_through("llm_assessment")

    assert state["llm_assessment"] == {"verdict": "revise"}
    assert state["status"] == "release_approve"


def test_replaying_an_unrecorded_node_names_what_is_available(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(index=0, node="ingest", update={"evidence": []})

    with pytest.raises(
        ValueError, match=r"no recorded step named 'publish'; recorded steps: \['ingest'\]"
    ):
        store.replay_through("publish")
