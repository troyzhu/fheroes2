#!/usr/bin/env bash
# Documentation gate for agent_play/docs.
#
# Per-file checks, all of which have caught real defects in this tree before:
#   1. The WRITING_STYLE contract (~/.claude/plugins/marketplaces/troyzhu/docs/WRITING_STYLE.md)
#   2. Wikilink resolution, since Obsidian links break silently on GitHub
#
# Tree-level fact checks, added after status prose went stale silently three times in one day
# (an index omitting its own sibling, "nothing here is implemented yet" surviving the training
# stack, a Built column claiming no learner exists). Form checks cannot catch a sentence whose
# facts moved, so facts are checked directly:
#   3. Index completeness: a moc README must mention every markdown sibling it indexes
#   4. Declared claims: any page may carry a verify block (exists/absent/grep, the same grammar
#      as verify_memory.sh), and status prose that can rot is expected to declare one, so the
#      moment reality flips, this gate fails instead of a reader being misled
#   5. Code paths: backticked src/ and python/ paths in non-archive pages must exist
#   6. Engine-surface completeness: every file changed under src/ relative to master must be
#      named in the inventory's engine-source ledger, so an unledgered engine touch cannot land
#
# Usage: agent_play/lint_docs.sh [file ...]     (no args lints the whole tree)
# Fact checks run only in whole-tree mode. Exits non-zero on any breach.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$HERE/docs"

FACTS=0
if [ "$#" -gt 0 ]; then
    FILES=("$@")
else
    # archive/sources holds fetched third-party files; they are provenance, not our writing.
    IFS=$'\n' read -r -d '' -a FILES < <(
        find "$DOCS" -name '*.md' -not -path '*/archive/sources/*' | sort && printf '\0'
    )
    FACTS=1
fi

FACTS="$FACTS" DOCS="$DOCS" python3 - "${FILES[@]}" <<'PY'
import os
import pathlib
import re
import subprocess
import sys

DOCS = pathlib.Path(os.environ["DOCS"]).resolve()
REPO = DOCS.parent.parent
files = [pathlib.Path(a).resolve() for a in sys.argv[1:]]

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
        if not s or s.startswith(("|", "#", "-", ">", "$", "---", "<!--")):
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

facts = []
if os.environ.get("FACTS") == "1":
    def rel(p):
        return str(p.relative_to(DOCS.parent.parent))

    # 3. Index completeness. A moc README indexes its own directory; the mapping below adds the
    # subdirectories whose files a parent README indexes. Mentioning the file's stem anywhere in
    # the README counts, so tables of backticked paths satisfy it as well as wikilinks.
    indexed_dirs = {DOCS / "research" / "works": DOCS / "research" / "README.md",
                    DOCS / "archive" / "experiments": DOCS / "archive" / "README.md"}
    for readme in sorted(DOCS.rglob("README.md")):
        head = readme.read_text(encoding="utf-8")[:400]
        if "type: moc" in head:
            indexed_dirs.setdefault(readme.parent, readme)
    for directory, readme in sorted(indexed_dirs.items()):
        if "sources" in directory.parts:
            continue
        text = readme.read_text(encoding="utf-8")
        for page in sorted(directory.glob("*.md")):
            if page == readme:
                continue
            if page.stem not in text:
                facts.append(f"{rel(readme)}: does not index sibling {page.name}")

    # 3b. Experiment scripts must be indexed in their README, so a script cannot silently fall
    # out of the record the way the first generation of scripts fell out of a temp directory.
    exp = DOCS.parent / "experiments"
    exp_readme = (exp / "README.md").read_text(encoding="utf-8") if (exp / "README.md").exists() else ""
    for script in sorted(exp.glob("*.py")):
        if script.name not in exp_readme:
            facts.append(f"experiments/README.md: does not index script {script.name}")

    # 3c. Library modules must be indexed in the library README, same rationale as 3b; the
    # gap was found when selfplay.py shipped without a row.
    lib = DOCS.parent.parent / "python" / "fheroes2_agent"
    lib_readme = (lib / "README.md").read_text(encoding="utf-8") if (lib / "README.md").exists() else ""
    for module in sorted(lib.glob("*.py")):
        if module.name != "__init__.py" and module.name not in lib_readme:
            facts.append(f"python/fheroes2_agent/README.md: does not index module {module.name}")

    # 4. Declared claims, the verify_memory.sh grammar.
    DECL = re.compile(r"<!--\s*verify\s*\n(.*?)-->", re.S)
    for f in files:
        for block in DECL.findall(f.read_text(encoding="utf-8")):
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                verb, _, rest = line.partition(" ")
                rest = rest.strip()
                if verb == "exists":
                    if not (REPO / rest).exists():
                        facts.append(f"{rel(f)}: exists {rest} -> missing")
                elif verb == "absent":
                    if (REPO / rest).exists():
                        facts.append(f"{rel(f)}: absent {rest} -> now present, update the claim's page")
                elif verb == "grep":
                    target, _, needle = rest.partition("::")
                    target, needle = target.strip(), needle.strip()
                    p = REPO / target
                    if not p.exists():
                        facts.append(f"{rel(f)}: grep {target} -> file missing")
                    elif needle not in p.read_text(encoding="utf-8", errors="replace"):
                        facts.append(f"{rel(f)}: grep {target} :: {needle!r} -> not found")
                else:
                    facts.append(f"{rel(f)}: unknown verify verb {verb!r}")

    # 5. Backticked code paths must exist. Archive pages are dated provenance and may name what
    # has since been deleted, so they are exempt. The prefixes are this repository's actual
    # layout, so a path inside another project's tree (src/Griddly/...) is not claimed.
    CODE_PATH = re.compile(r"`((?:src/(?:fheroes2|engine|agent_\w+|dist|thirdparty)|python)/[A-Za-z0-9_./-]+)`")
    for f in files:
        if "archive" in f.parts:
            continue
        text = f.read_text(encoding="utf-8")
        for path in sorted(set(CODE_PATH.findall(text))):
            if not (REPO / path.rstrip("/")).exists():
                facts.append(f"{rel(f)}: names `{path}`, which does not exist")

    # 6. Engine-surface completeness. Every file changed under src/ relative to master must be
    # matched in the inventory's ledger section, by path, by directory, or by stem (which is how
    # a brace form like screen.{h,cpp} matches screen.h). An unledgered engine touch fails here.
    inventory = DOCS / "implementation" / "inventory.md"
    section = re.search(r"## Engine-source surface.*?(?=\n## )", inventory.read_text(encoding="utf-8"), re.S)
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "master", "--name-only", "--", "src/"],
                          capture_output=True, text=True)
    if section is None:
        facts.append("inventory.md: engine-source surface section not found")
    elif diff.returncode != 0:
        print("  note  engine-surface check skipped, git diff against master failed")
    else:
        ledger = section.group(0)
        for changed in sorted(filter(None, diff.stdout.splitlines())):
            p = pathlib.PurePosixPath(changed)
            # A directory reference counts only when the ledger names the directory itself,
            # not when the directory is merely the prefix of some other file's full path.
            named = (changed in ledger) or (p.stem + "." in ledger) \
                or re.search(re.escape(str(p.parent) + "/") + r"(?![A-Za-z0-9_])", ledger)
            if not named:
                facts.append(f"inventory.md: engine change {changed} is not in the ledger")

print()
if facts:
    for x in facts:
        print("  FACT  " + x)
    print()
if failed or facts:
    print("lint_docs: BREACHES FOUND")
    sys.exit(1)
print(f"lint_docs: {len(files)} files clean, facts checked")
PY
