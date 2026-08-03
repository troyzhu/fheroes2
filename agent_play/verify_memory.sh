#!/usr/bin/env bash
# Check that this project's agent memory still describes reality.
#
# Memory files decay silently: they are prose, they live outside the repository, and nothing
# fails when a claim stops being true. This ran once by hand and found a line instructing a
# future session to "correct" the notation into the wrong convention.
#
# The mechanism is that each memory file declares what makes it true, in an HTML comment so it
# stays invisible when the file is read:
#
#   <!-- verify
#   exists  agent_play/docs/overview.md
#   absent  agent_play/docs/notation.md
#   grep    CLAUDE.md :: rl/ against
#   -->
#
# Every backticked repository path anywhere in a memory file is checked too, without needing a
# declaration, since that is the class that rots most often.
#
# Usage: agent_play/verify_memory.sh
# Exits non-zero on any failed claim.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Claude Code mangles the project path by replacing "/" and " " with "-".
MEM="$HOME/.claude/projects/$(printf '%s' "$REPO" | tr '/ ' '--')/memory"

if [ ! -d "$MEM" ]; then
    echo "verify_memory: no memory directory at $MEM"
    echo "verify_memory: nothing to check"
    exit 0
fi

REPO="$REPO" MEM="$MEM" python3 <<'PY'
import os
import pathlib
import re
import sys

REPO = pathlib.Path(os.environ["REPO"])
MEM = pathlib.Path(os.environ["MEM"])

DECL = re.compile(r"<!--\s*verify\s*\n(.*?)-->", re.S)
# Backticked repo-relative paths. Brace and glob forms are skipped as unresolvable.
PATH_IN_PROSE = re.compile(r"`((?:agent_play|src|python|configs)/[A-Za-z0-9_./-]+)`")

failures, checks = [], 0
files = sorted(MEM.glob("*.md"))

for f in files:
    text = f.read_text(encoding="utf-8")

    # 1. Declared claims.
    for block in DECL.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            verb, _, rest = line.partition(" ")
            rest = rest.strip()
            checks += 1
            if verb == "exists":
                if not (REPO / rest).exists():
                    failures.append(f"{f.name}: exists {rest} -> missing")
            elif verb == "absent":
                if (REPO / rest).exists():
                    failures.append(f"{f.name}: absent {rest} -> still present")
            elif verb == "grep":
                target, _, needle = rest.partition("::")
                target, needle = target.strip(), needle.strip()
                p = REPO / target
                if not p.exists():
                    failures.append(f"{f.name}: grep {target} -> file missing")
                elif needle not in p.read_text(encoding="utf-8", errors="replace"):
                    failures.append(f"{f.name}: grep {target} :: {needle!r} -> not found")
            else:
                failures.append(f"{f.name}: unknown verb {verb!r}")

    # 2. Every repo path named in prose.
    for path in sorted(set(PATH_IN_PROSE.findall(text))):
        checks += 1
        if not (REPO / path.rstrip("/")).exists():
            failures.append(f"{f.name}: path `{path}` does not exist")

    # 3. Cross-links between memory files.
    for link in sorted(set(re.findall(r"\[\[([^\]]+)\]\]", text))):
        checks += 1
        if not (MEM / f"{link.split('|')[0].strip()}.md").exists():
            failures.append(f"{f.name}: [[{link}]] -> no such memory")

# 4. The index and the files agree.
index = MEM / "MEMORY.md"
if index.exists():
    listed = set(re.findall(r"\]\(([A-Za-z0-9._-]+\.md)\)", index.read_text(encoding="utf-8")))
    actual = {p.name for p in files} - {"MEMORY.md"}
    for missing in sorted(actual - listed):
        failures.append(f"MEMORY.md: {missing} exists but is not indexed")
    for ghost in sorted(listed - actual):
        failures.append(f"MEMORY.md: indexes {ghost}, which does not exist")
    checks += len(actual | listed)

print(f"verify_memory: {len(files)} memory files, {checks} claims checked")
if failures:
    print()
    for x in failures:
        print("  FAIL  " + x)
    print("\nverify_memory: STALE CLAIMS FOUND")
    sys.exit(1)
print("verify_memory: all claims hold")
PY
