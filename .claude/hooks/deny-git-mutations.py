#!/usr/bin/env python3
"""PreToolUse hook denying destructive git invocations and shell wrappers.

Two independent detection layers (a defanged substring scan and a structural
shlex tokenization) — either may flag a command.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

# === Single source of truth =================================================
# Git verbs that ALWAYS mutate (repo, index, working tree, or remote).
ALWAYS_DESTRUCTIVE: frozenset[str] = frozenset(
    {
        "add",
        "am",
        "apply",
        "bundle",
        "checkout",
        "checkout-index",
        "cherry-pick",
        "clean",
        "clone",
        "commit",
        "commit-graph",
        "commit-tree",
        "credential",
        "daemon",
        "fast-import",
        "fetch",
        "fetch-pack",
        "filter-branch",
        "gc",
        "hash-object",
        "hook",
        "http-backend",
        "imap-send",
        "index-pack",
        "init",
        "maintenance",
        "merge",
        "mergetool",
        "mktag",
        "mktree",
        "multi-pack-index",
        "mv",
        "p4",
        "pack-objects",
        "pack-refs",
        "prune",
        "prune-packed",
        "pull",
        "push",
        "read-tree",
        "rebase",
        "repack",
        "replay",
        "request-pull",
        "reset",
        "restore",
        "revert",
        "rm",
        "scalar",
        "send-email",
        "send-pack",
        "svn",
        "switch",
        "unpack-objects",
        "update-index",
        "update-ref",
        "update-server-info",
        "write-tree",
    }
)

# Read-only flags for mixed verbs whose presence overrides the "positional ⇒ create"
# heuristic (e.g. `git tag -v v1` is verify, not create).
READ_ONLY_FLAGS_PER_VERB: dict[str, frozenset[str]] = {
    "branch": frozenset({"-v", "-vv", "--verbose", "-r", "-a", "--list", "--show-current", "-q", "--quiet"}),
    "tag": frozenset({"-v", "--verify", "-l", "--list", "-n", "-n1", "-n2", "-n3"}),
}

# Mixed verbs: deny only when a mutating flag is present.
MUTATING_FLAGS: dict[str, frozenset[str]] = {
    "branch": frozenset(
        {
            "-d",
            "-D",
            "--delete",
            "-m",
            "-M",
            "--move",
            "-c",
            "-C",
            "--copy",
            "-u",
            "--set-upstream",
            "--set-upstream-to",
            "--unset-upstream",
            "--edit-description",
        }
    ),
    "tag": frozenset(
        {
            "-a",
            "-s",
            "-u",
            "-d",
            "--delete",
            "-f",
            "--force",
            "-m",
            "-F",
            "--annotate",
            "--sign",
            "--local-user",
            "--message",
            "--file",
        }
    ),
    "config": frozenset(
        {
            "--set",
            "--add",
            "--append",
            "--replace-all",
            "--unset",
            "--unset-all",
            "--rename-section",
            "--remove-section",
            "-e",
            "--edit",
        }
    ),
    "replace": frozenset({"-d", "--delete", "--edit", "--graft", "--convert-graft-file", "-f", "--force"}),
    "symbolic-ref": frozenset({"-d", "--delete"}),
}

# Mixed verbs: deny only when the first positional matches a mutating subcommand.
MUTATING_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "notes": frozenset({"add", "copy", "append", "edit", "remove", "prune", "merge"}),
    "remote": frozenset({"add", "rename", "remove", "rm", "set-head", "set-branches", "set-url", "prune", "update"}),
    "submodule": frozenset(
        {"add", "init", "update", "deinit", "sync", "set-branch", "set-url", "absorbgitdirs", "foreach"}
    ),
    "worktree": frozenset({"add", "remove", "move", "prune", "repair", "lock", "unlock"}),
    "reflog": frozenset({"expire", "delete", "drop", "write"}),
    "bisect": frozenset({"start", "good", "bad", "new", "old", "skip", "reset", "replay", "run", "terms"}),
    "sparse-checkout": frozenset({"init", "set", "add", "reapply", "disable"}),
    "rerere": frozenset({"clear", "forget", "gc"}),
}

# Verbs where the bare form mutates but specific subcommands are read-only.
# Default = DENY; allow only when the first positional is a read-only subcommand.
DENY_UNLESS_READ_ONLY_SUB: dict[str, frozenset[str]] = {
    "stash": frozenset({"list", "show", "create"}),
}
RO_FLAGS_TAKING_VALUE: frozenset[str] = frozenset(
    {
        "--contains",
        "--no-contains",
        "--merged",
        "--no-merged",
        "--points-at",
        "--format",
        "--sort",
        "--column",
        "--color",
    }
)

# Shell wrappers whose argument content cannot be reliably introspected.
SHELL_WRAPPERS: tuple[str, ...] = ("bash", "sh", "zsh", "dash", "fish", "ksh")

# `git`-level options that take a value (positional or `=value`).
GIT_OPTS_WITH_VALUE: frozenset[str] = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--exec-path",
        "--super-prefix",
    }
)
GIT_OPTS_STANDALONE: frozenset[str] = frozenset(
    {
        "--bare",
        "--no-replace-objects",
        "--literal-pathspecs",
        "--no-optional-locks",
        "--paginate",
        "-p",
        "--no-pager",
    }
)


# === Helpers =================================================================
def strip_git_level_options(tokens: list[str]) -> list[str]:
    """Drop `git -C path`, `--git-dir=foo`, `-c key=val`, etc. before the verb."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in GIT_OPTS_WITH_VALUE:
            i += 2
        elif (
            tok in GIT_OPTS_STANDALONE
            or any(tok.startswith(opt + "=") for opt in GIT_OPTS_WITH_VALUE)
            or tok.startswith("-c")
            and tok != "-c"
        ):
            i += 1
        else:
            break
    return tokens[i:]


def first_positional(args: list[str]) -> str | None:
    """First non-flag, non-flag-value token."""
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in RO_FLAGS_TAKING_VALUE:
            skip_next = True
            continue
        if not arg.startswith("-"):
            return arg
    return None


def check_git_verb(verb: str, args: list[str]) -> str | None:
    """Return a deny reason if `git <verb> <args>` mutates, else None."""
    if verb in ALWAYS_DESTRUCTIVE:
        return f"`git {verb}` is destructive"

    if verb in MUTATING_FLAGS:
        # Match exact flag tokens AND `--flag=value` prefix tokens.
        arg_set = set(args)
        bad_flag: str | None = None
        for mflag in MUTATING_FLAGS[verb]:
            if mflag in arg_set:
                bad_flag = mflag
                break
            if mflag.startswith("--") and any(a.startswith(mflag + "=") for a in args):
                bad_flag = mflag
                break
        if bad_flag:
            return f"`git {verb} {bad_flag}` mutates"
        # Read-only flags shut off the create-form heuristic for branch/tag.
        ro_flags = READ_ONLY_FLAGS_PER_VERB.get(verb, frozenset())
        if arg_set & ro_flags:
            return None
        # branch/tag: any positional argument means create-form.
        if verb in {"branch", "tag"} and first_positional(args) is not None:
            return f"`git {verb} <name>` is a create/write form"
        # config: two positionals = implicit set; new-style verbs.
        if verb == "config":
            # Read-only flags that take values must not count those values as
            # positionals (--get-urlmatch <section> <url> is read-only).
            if any(a == "--get-urlmatch" or a.startswith("--get") for a in args):
                return None
            positionals = [a for a in args if not a.startswith("-")]
            if len(positionals) >= 2:
                return "`git config <key> <value>` writes config"
            if positionals and positionals[0] in {
                "set",
                "unset",
                "edit",
                "rename-section",
                "remove-section",
            }:
                return f"`git config {positionals[0]}` writes config"
        if verb == "replace" and first_positional(args) is not None and not ({"-l", "--list"} & set(args)):
            return "`git replace <obj> ...` creates a replace ref"
        if verb == "symbolic-ref":
            positionals = [a for a in args if not a.startswith("-")]
            if len(positionals) >= 2:
                return "`git symbolic-ref <name> <ref>` writes a symref"
        return None

    if verb in MUTATING_SUBCOMMANDS:
        sub = first_positional(args)
        if sub and sub in MUTATING_SUBCOMMANDS[verb]:
            return f"`git {verb} {sub}` mutates"

    if verb in DENY_UNLESS_READ_ONLY_SUB:
        sub = first_positional(args)
        if sub is None or sub not in DENY_UNLESS_READ_ONLY_SUB[verb]:
            return f"`git {verb}` mutates (only read-only subcommands allowed)"

    return None


def check_segment(tokens: list[str]) -> str | None:
    """Inspect a single shell command segment (post split on &&/;/|/etc.)."""
    # Skip leading `FOO=bar`, `command`, `exec`.
    idx = 0
    while idx < len(tokens) and (
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]) or tokens[idx] in {"command", "exec"}
    ):
        idx += 1
    if idx >= len(tokens):
        return None

    head = tokens[idx].lstrip("\\")  # `\git` -> `git`
    rest = tokens[idx + 1 :]

    if head in SHELL_WRAPPERS and rest and rest[0] == "-c":
        return f"`{head} -c` wrapper hides the inner command"
    if head in {"eval", "xargs"}:
        return f"`{head}` hides the target command"
    if head == "find" and ("-exec" in rest or "-delete" in rest):
        return "`find -exec`/`-delete` hides the target command"
    if head != "git":
        return None

    after_globals = strip_git_level_options(rest)
    if not after_globals:
        return None
    return check_git_verb(after_globals[0], after_globals[1:])


# === Layer 1: defanged substring scan ========================================
def substring_scan(command: str) -> str | None:
    """Catches wrapper bypasses and quote/escape obfuscation by stripping
    quotes+backslashes+punctuation and regex-scanning the residue.
    """
    flat = command.lower().translate(str.maketrans("", "", "\"'`\\"))
    # Replace list/call punctuation with spaces so e.g. ['git','stash'] flattens
    # to ` git  stash ` for the same regex match as plain `git stash`.
    flat = re.sub(r"[,\[\]\(\)]", " ", flat)
    flat = re.sub(r"\s+", " ", flat)

    for wrapper in SHELL_WRAPPERS:
        if re.search(rf"\b{wrapper}\s+-c\b", flat):
            return f"`{wrapper} -c` wrapper detected"
    if re.search(r"\beval\b", flat):
        return "`eval` wrapper detected"
    if re.search(r"\bxargs\b", flat):
        return "`xargs` detected"
    if re.search(r"\bfind\b[^;|&]*-(exec|delete)\b", flat):
        return "`find -exec`/`-delete` detected"

    # ALWAYS-destructive verbs plus `stash` (subcommand-aware; only deny if the
    # next token isn't a read-only stash subcommand).
    layer1_verbs = ALWAYS_DESTRUCTIVE | set(DENY_UNLESS_READ_ONLY_SUB)
    verb_alt = "|".join(re.escape(v) for v in sorted(layer1_verbs, key=len, reverse=True))
    # Trailing `(?![-\w])` ensures `merge` doesn't match `merge-base`.
    git_destructive = (
        r"\bgit\b"
        r"(?:\s+-[A-Za-z]+(?:=\S+)?|\s+--[A-Za-z][\w-]*(?:=\S+)?)*"
        rf"\s+({verb_alt})(?![-\w])\s*([\w-]*)"
    )
    match = re.search(git_destructive, flat)
    if match:
        verb, next_tok = match.group(1), match.group(2)
        if verb in DENY_UNLESS_READ_ONLY_SUB and next_tok in DENY_UNLESS_READ_ONLY_SUB[verb]:
            return None
        return f"`git {verb}` is destructive"
    return None


# === Layer 2: structural tokenization =======================================
SEGMENT_SEPARATORS: frozenset[str] = frozenset(
    {
        "&&",
        "||",
        ";",
        "|",
        "|&",
        "&",
        "(",
        ")",
        "{",
        "}",
    }
)


def preprocess(command: str) -> str:
    """Make separators easy for shlex by surrounding them with spaces."""
    out = command.replace("\n", " ; ")
    for op in ("&&", "||", "|&", ";", "|"):
        out = out.replace(op, f" {op} ")
    return out


def structural_scan(command: str) -> str | None:
    try:
        tokens = shlex.split(preprocess(command), comments=False, posix=True)
    except ValueError as exc:
        return f"unparseable command: {exc}"

    segment: list[str] = []
    for tok in tokens:
        if tok in SEGMENT_SEPARATORS:
            reason = check_segment(segment) if segment else None
            if reason:
                return reason
            segment = []
        else:
            segment.append(tok)
    return check_segment(segment) if segment else None


# === Entry point ============================================================
def main() -> None:
    payload = json.load(sys.stdin)
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = substring_scan(command) or structural_scan(command)
    if reason:
        json.dump(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked by deny-git-mutations: {reason}. ",
                }
            },
            sys.stdout,
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
