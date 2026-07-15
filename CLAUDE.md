# CLAUDE.md

Guidance for AI agents working in this repo. These are **non-negotiable** engineering principles —
check every change against them. They exist because violations recur; treat them as review gates,
not suggestions.

## Core principles

1. **Fail fast, no defensive fallbacks.** For internal invariants, use strict access (`d[key]`,
   direct attribute) and let it raise. Do NOT add `.get()`/`if x is None`/`try-except` that swallows
   or silently continues — that masks bugs the crash would have surfaced. Fallbacks are legitimate
   ONLY for genuine external/untrusted input (map clicks, DEM lookups, file/network data, env vars).
   If you catch yourself writing a guard "just in case" around your own code's data, delete it and
   let it fail loud.

2. **Single source of truth — no drift.** Never maintain two pieces of state that must agree (a bool
   plus the value it's derived from), or a hand-maintained list/set that must track an enum. Derive
   from the data, collapse to one field, or assert the relationship at import time.

3. **Fix the root cause, not the symptom.** When the same class of failure appears at multiple sites,
   stop patching each read/call site. Find where the bad state is *produced* and fix it there. One
   correct emitter beats N defensive readers.

4. **No duplicated logic — extract and share.** If the same block appears in 2+ places, factor it into
   one function and call it. Duplicated logic drifts and gets fixed in only some copies.

5. **Prefer the existing pattern.** Before adding a new approach, find how the codebase already solves
   the same problem and follow it (e.g. reload-safe enum handling via `enum_eq`/`.name`, commit-time
   node materialisation, import-time bijection asserts). Consistency over novelty.

## Style

- Terse, high-signal docstrings and comments — explain the *why*, not the obvious *what*. No
  restating the code in prose. Max 2 or 3 liens of comments and docstring text blocks, but still Args in Google style.
- Explicit `if/else` blocks over nested/clever ternaries when there's real branching logic.
- Match the surrounding code's naming, structure, and comment density.

## Workflow

- Tests, `ruff`, and `mypy` must stay green; coverage gate must be met. Run the suite before declaring done.
- When you change behaviour, update the tests that encoded the old behaviour AND add a regression test
  for the specific bug — don't just make existing tests pass.
- Logging level policy and the `SKIRESORT_LOG_LEVEL` knob live in `skiresort_planner/logging_setup.py`;
  per-rerun/click detail is DEBUG, milestones are INFO. Don't add INFO spam to hot paths.
