"""Persist each workflow node's output so a run can be inspected and replayed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STEP_FILE = re.compile(r"^(\d{2})-([a-z_]+)\.json$")


@dataclass(frozen=True)
class RunStep:
    """One node's contribution to the run, as it was written to disk."""

    index: int
    node: str
    update: dict[str, Any]

    @property
    def filename(self) -> str:
        return f"{self.index:02d}-{self.node}.json"


class RunArtifactStore:
    """Write and read the per-node record of a single run.

    A LangGraph checkpoint resumes a graph; these files serve the other need. They
    stay readable, survive a code change that would invalidate a serialized
    checkpoint, and let one slow step be re-run from the step before it rather than
    from the top of a forty-minute pipeline.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, *, index: int, node: str, update: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        step = RunStep(index=index, node=node, update=update)
        path = self.directory / step.filename
        path.write_text(json.dumps(update, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def steps(self) -> tuple[RunStep, ...]:
        """Every recorded step, ordered by the index in its filename."""

        if not self.directory.is_dir():
            return ()
        found: list[RunStep] = []
        for path in self.directory.iterdir():
            match = _STEP_FILE.match(path.name)
            if match is None:
                continue
            found.append(
                RunStep(
                    index=int(match.group(1)),
                    node=match.group(2),
                    update=json.loads(path.read_text(encoding="utf-8")),
                )
            )
        return tuple(sorted(found, key=lambda step: step.index))

    def replay_through(self, node: str) -> tuple[dict[str, Any], str]:
        """Merge every step up to and including the last run of ``node``.

        Returns the accumulated state and the node name to resume as, so the graph
        continues with whatever follows that node.
        """

        steps = self.steps()
        matching = [step for step in steps if step.node == node]
        if not matching:
            recorded = sorted({step.node for step in steps})
            raise ValueError(f"no recorded step named {node!r}; recorded steps: {recorded}")

        last = matching[-1]
        state: dict[str, Any] = {}
        for step in steps:
            if step.index > last.index:
                break
            state.update(step.update)
        return state, node
