#!/usr/bin/env bash
# Documentation gate for agent_play/docs.
#
# Two checks, both of which have caught real defects in this tree before:
#   1. The WRITING_STYLE contract (~/.claude/plugins/marketplaces/troyzhu/docs/WRITING_STYLE.md)
#   2. Wikilink resolution, since Obsidian links break silently on GitHub
#
# Usage: agent_play/lint_docs.sh [file ...]     (no args lints the whole tree)
# Exits non-zero on any breach.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$HERE/docs"

if [ "$#" -gt 0 ]; then
    FILES=("$@")
else
    # archive/sources holds fetched third-party files; they are provenance, not our writing.
    IFS=$'\n' read -r -d '' -a FILES < <(
        find "$DOCS" -name '*.md' -not -path '*/archive/sources/*' | sort && printf '\0'
    )
fi

python3 - "${FILES[@]}" <<'PY'
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent / "docs" if False else None
files = [pathlib.Path(a).resolve() for a in sys.argv[1:]]
root = pathlib.Path(__file__).resolve()

BANNED = re.compile(
    r"not just [^,]+, but|isn't just|more than just|here's (the thing|why|how)|"
    r"let's dive|dive into|delve|the real question is|here's where it gets|"
    r"^\s*(crucially|importantly|notably|it's worth noting)\b|"
    r"you might wonder|you may be asking|game-changing|"
    r"unlock(s|ing)? the|harness(es|ing)? the (power|potential)|"
    r"navigate the landscape|in today's fast-paced",
    re.I | re.M,
)

# Thresholds from the WRITING_STYLE contract.
MAX_DASH_PER_1K_CHARS = 1.0
MAX_BOLD_PER_1K_WORDS = 6.0
MAX_PARA_WORDS = 160
WRAP_WIDTH = 100  # a prose line shorter than this, adjacent to another, reads as hard-wrapped


def strip_math_and_quotes(text):
    out, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if s == "$$":
            in_block = not in_block
            continue
        if in_block or s.startswith(">"):
            continue
        out.append(line)
    joined = "\n".join(out)
    joined = re.sub(r"\$\$[^$]*\$\$", "", joined)
    joined = re.sub(r"\$[^$]*\$", "", joined)
    return joined


def strip_fenced_code(text):
    """Drop ``` blocks entirely; they may contain blank lines and would otherwise
    be split into fragments that look like hard-wrapped prose."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def paragraphs(text):
    """Prose paragraphs only, with math blocks, tables, lists, and code skipped."""
    in_block = False
    for para in re.split(r"\n\s*\n", strip_fenced_code(text)):
        s = para.strip()
        if in_block:
            if s.endswith("$$"):
                in_block = False
            continue
        if s.startswith("$$"):
            if s == "$$" or not s.endswith("$$"):
                in_block = True
            continue
        if not s or s.startswith(("|", "#", "-", ">", "$", "---")):
            continue
        if re.match(r"^\d+[.)]\s", s):  # ordered list
            continue
        yield s


def headings_of(path):
    """Normalized heading text of a file, for anchor resolution."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {
        re.sub(r"\s+", " ", h.strip().lower())
        for h in re.findall(r"^#{1,6}\s+(.+?)\s*$", src, re.M)
    }


def resolve_wikilinks(path, text):
    """Return unresolved [[target]] links, checking both the file and the heading anchor.

    An anchor that does not name a real heading is the failure mode a file rename or a
    section retitle produces, and it is invisible on GitHub, so it is checked here."""
    bad = []
    for raw in re.findall(r"\[\[([^\]]+)\]\]", text):
        # Inside a table cell the alias pipe is escaped as \| so the row does not split.
        target = raw.replace("\\|", "|").split("|", 1)[0]
        stem, _, anchor = target.partition("#")
        stem = stem.strip()
        if not stem:
            cand = path  # [[#heading]], an intra-file anchor
        else:
            cand = (path.parent / stem).resolve()
            if cand.suffix != ".md":
                cand = cand.with_suffix(".md")
            if not cand.exists():
                bad.append(raw)
                continue
        if anchor and re.sub(r"\s+", " ", anchor.strip().lower()) not in headings_of(cand):
            bad.append(raw + "  (no such heading)")
    return bad


rows, failed = [], False
for f in files:
    text = f.read_text(encoding="utf-8")
    prose = strip_math_and_quotes(text)

    chars = len(prose)
    words = len(prose.split())
    dash = prose.count("—") * 1000 / chars if chars else 0.0
    bold = len(re.findall(r"\*\*[^*]+\*\*", prose)) * 1000 / words if words else 0.0
    banned = len(BANNED.findall(prose))
    qhead = len(re.findall(r"^#{1,6}\s.*\?\s*$", text, re.M))
    longp = sum(1 for p in paragraphs(text) if len(p.split()) > MAX_PARA_WORDS)
    wrapped = sum(
        1
        for p in paragraphs(text)
        if "\n" in p and all(len(ln) < WRAP_WIDTH for ln in p.splitlines())
    )
    links = resolve_wikilinks(f, text)

    problems = []
    if dash > MAX_DASH_PER_1K_CHARS:
        problems.append(f"em-dash {dash:.2f}/1k")
    if bold > MAX_BOLD_PER_1K_WORDS:
        problems.append(f"bold {bold:.2f}/1k")
    if banned:
        problems.append(f"banned x{banned}")
    if qhead:
        problems.append(f"question-heading x{qhead}")
    if longp:
        problems.append(f"para>{MAX_PARA_WORDS}w x{longp}")
    if wrapped:
        problems.append(f"hard-wrapped x{wrapped}")
    if links:
        problems.append("dead links: " + ", ".join(links))

    name = str(f).split("agent_play/", 1)[-1]
    if problems:
        failed = True
        rows.append(f"  FAIL  {name}\n          " + "\n          ".join(problems))
    else:
        rows.append(f"  ok    {name}")

print("\n".join(rows))
print()
if failed:
    print("lint_docs: BREACHES FOUND")
    sys.exit(1)
print(f"lint_docs: {len(files)} files clean")
PY
