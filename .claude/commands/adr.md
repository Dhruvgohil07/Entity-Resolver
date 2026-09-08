---
description: Record a decision in the CLAUDE.md Open questions log
argument-hint: <the decision, or the open question it settles>
---

Record this in CLAUDE.md:

$ARGUMENTS

The decisions log for this project is `### Open questions`, inside `## Project status`. There is
no separate Decisions Log section, and you must not create one — CLAUDE.md's commit convention
says so explicitly.

That section has two parts:

1. An unlabelled list at the top — judgment calls that are made but not yet backed by evidence.
2. `Settled by measurement, recorded so it is not re-litigated:` — decisions with numbers behind them.

Pick the right action:

- **Settled by evidence** -> add a bullet under "Settled by measurement".
- **A judgment call with no measurement yet** -> add a bullet to the top list, and state what
  evidence would settle it.
- **Evidence that resolves something already in the top list** -> move that bullet down into
  "Settled by measurement" and rewrite it with the numbers. Do not leave a copy behind.

Write in the prose style already used there — flowing sentences, not labelled fields:

- Open with a bold sentence stating the decision as a claim, in the style of
  "**`'` folds to inches, not feet.**"
- Then the alternative and why it lost. If this reverses an earlier decision, say so, and say what
  the earlier reasoning was.
- Then the evidence, with real numbers — counts, F1s, file names, measured distributions.
  "Cleaner" and "better" are not reasons. If that is all there is, the entry belongs in the top
  list, not the settled one.
- End with the condition that would overturn it ("Revisit only if...").

Wrap at 100 characters to match the file. Keep the entry under ten lines.

Do not commit. CLAUDE.md's convention is that this lands in the same commit as the code that
settles it, with the commit body saying so.
