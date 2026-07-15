---
name: check-patterns
description: Audit the current change against the non-negotiable engineering principles in CLAUDE.md and fix every violation. Use when the user invokes /check-patterns, or asks to review/clean the diff against the project's coding rules ("check patterns", "check against claude.md", "did you follow the rules").
---

# check-patterns

Audit the working change against the **Core principles** in the repo's `CLAUDE.md` and fix every
violation found. This is a self-review gate so the user doesn't have to hand-catch the same recurring
mistakes.

## Steps

1. **Load the rules.** Read `CLAUDE.md` at the repo root. The numbered "Core principles" + "Style"
   sections are the checklist. Do not rely on memory — read the file so you check the current wording.

2. **Get the diff to audit.** Run `git diff` (unstaged), `git diff --staged`, and `git diff main...HEAD`
   as appropriate to capture everything changed on this branch/turn. If the user names specific files,
   scope to those. Only audit changed lines — don't re-review the whole repo.

3. **Grep for the mechanical smells** across the changed files (fast first pass):
   - Defensive fallbacks on internal data: `\.get(` on graph/context dicts, `if .* is None`,
     `try:` blocks that swallow, `-> .* | None` returns whose callers just raise anyway.
   - Drift risk: two fields assigned from the same condition; a hardcoded set/list of enum names.
   - Duplication: the same multi-line block appearing in 2+ changed files.
   - Unreadble long comment and docstring blocks

4. **Read each candidate in context** and classify against the principles. Be precise, not trigger-happy:
   - A `.get()` on genuine external input (a map-click id, DEM value, file field) is CORRECT — leave it.
   - A `.get()`+explicit `raise` with a clear message is already fail-fast — leave it.
   - Only flag guards around the code's *own* invariants.

5. **Fix every real violation** in the working tree (strict access, remove the log-before-raise,
   collapse drift to one source, extract duplication, move a symptom-patch to its root emitter).
   For anything ambiguous or a judgment call (e.g. "is this input external?"), do NOT silently
   change it — list it and ask.

6. **Verify:** run `ruff check`, `mypy`, and the test suite. Update any test that encoded the old
   (violating) behaviour and add a regression test where a bug was fixed.

7. **Report** as a table: file:line — which principle — what you changed (or why you left it). End
   with the ruff/mypy/test result.

## Notes
- Prefer the codebase's existing pattern over inventing a new one (principle 6): before "fixing"
  something, grep for how the repo already handles the same case.
- Keep it scoped to the current change unless the user explicitly asks for a whole-repo sweep.
