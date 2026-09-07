---
description: Append a Decisions Log entry to CLAUDE.md
argument-hint: <short decision title>
---

Append a numbered entry to the `## Decisions Log` section of CLAUDE.md for this decision:

$ARGUMENTS

Create the section if it does not exist yet, placing it directly before `## Commands`.

Entry format — keep it to six lines or fewer:

```
### ADR-<n> — <title>

<date>. <The decision, one or two sentences, stated as a choice made.>
**Rejected:** <the alternative and the specific reason it lost.>
**Revisit if:** <the observation that would overturn this.>
```

Rules:

- Number sequentially from the highest existing ADR. Never renumber existing entries.
- Use today's date in ISO form.
- The reason must be concrete — a measured number, a wheel that does not exist, a cost
  asymmetry. "Cleaner" or "better" is not a reason; if that is all there is, ask for the real one.
- If the decision is already covered by an existing ADR, amend that entry instead of adding a
  near-duplicate, and say which one you amended.
- Do not commit. The commit convention in CLAUDE.md expects the implementing commit to reference
  the ADR, so the entry lands first and the code cites it.
