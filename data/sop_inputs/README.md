# SOP inputs

Drop SOP files here and point the local runner at one:

```powershell
$env:LOCAL_LLM_MODEL = "qwen/qwen3.8-27b"
uv run python examples/run_local.py "data/sop_inputs/<file>.pdf"
```

Supported: `.txt`, `.md`, and `.pdf`. PDF reading needs the `documents` extra
(`uv sync --extra documents`); scanned image-only PDFs are rejected rather than
silently producing empty evidence.

## Why the contents are untracked

Everything in this folder except this README is git-ignored. Real SOPs carry company
names, staff names, and business thresholds, and their licensing is usually unresolved
— `docs/CLEAN_ROOM.md` withholds exactly that class of input from the public
repository. Synthetic SOPs that belong in the repository live in `tests/helpers.py`
and `examples/run_local.py`.
