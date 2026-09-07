---
description: Lint and test in the 3.12 venv
allowed-tools: Bash(.venv/Scripts/python:*)
---

Lint result:

!`.venv/Scripts/python -m ruff check src tests`

Test result:

!`.venv/Scripts/python -m pytest -q`

If either command above reports a failure, fix the underlying code — do not adjust a test to
make it pass, and do not silence a ruff rule without saying why in the same edit. If both are
clean, reply with the passing test count and nothing else.

Note the interpreter: this project's `.venv` is Python 3.12, not the machine default. Always
invoke it as `.venv/Scripts/python -m <tool>` rather than bare `pytest` or `ruff`.
