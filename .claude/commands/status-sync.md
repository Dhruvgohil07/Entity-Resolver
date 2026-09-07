---
description: Reconcile the CLAUDE.md Project status block with the actual tree
allowed-tools: Bash(.venv/Scripts/python:*), Bash(git log:*), Bash(find:*)
---

Current suite:

!`.venv/Scripts/python -m pytest -q --tb=no`

Update the `## Project status` section of CLAUDE.md so it matches reality:

1. Correct the passing-test count from the run above (it appears twice — the status block and
   the `## Commands` block).
2. For each directory under `src/dedup/`, determine whether it is implemented or still an empty
   `__init__.py`, and make the status text say so accurately. A module with code but no tests is
   "implemented, untested" — say that rather than listing it alongside tested modules.
3. State what the next work is, based on what is actually blocking: a stage cannot be evaluated
   before the stage it depends on has data flowing through it.
4. Check whether the example commands in `## Commands` still exist. Move anything that no longer
   runs into the "Not built yet" block, and promote anything that now works out of it.

Constraints:

- Touch only `## Project status`, `### Open questions`, and `## Commands`. Leave the pipeline
  architecture, invariants, stack, and commit-convention sections alone.
- Do not delete an open question because the code changed — resolve it explicitly or leave it.
- Do not commit.
