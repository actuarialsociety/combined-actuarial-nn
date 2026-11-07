#!/usr/bin/env python3
"""Turn citation markers like [1] or [1, 2] in a lesson into links to the papers.

Each lesson's "## Learn more" slide is a numbered reference list; the first link of
item N (its DOI or paper page) becomes the target of every [N] in the lesson body, so a
reader can click through from a model's name to the academic source. Markers whose
numbers are not all in the list (e.g. layer sizes [20, 15, 10]) are left alone, as are
code, math and the reference list itself. Idempotent.

    python3 scripts/link_citations.py            # rewrite lessons/*.md in place
    python3 scripts/link_citations.py --check    # exit 1 if a citation is not linked (CI)
    python3 scripts/link_citations.py --list     # show every marker and what it maps to
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_RE = re.compile(r"^(\d+)\.\s.*?\]\((https?://[^)\s]+)\)", re.MULTILINE)
# [1] or [1, 2]: not part of a link label "[..](" and not a wiki link "[[..]]"
CITE_RE = re.compile(r"(?<![\[\w])\[(\d+(?:,\s*\d+)*)\](?![(\[\]])")
SKIP_RE = re.compile(r"```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$|<!--.*?-->", re.DOTALL)


def references(text: str) -> dict[str, str]:
    m = re.search(r"^## Learn more\s*$", text, re.MULTILINE)
    return dict(REF_RE.findall(text[m.end():])) if m else {}


def link_body(body: str, refs: dict[str, str], found: list | None = None) -> str:
    def cite(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        if not all(n in refs for n in nums):
            return m.group(0)
        if found is not None:
            found.append(m.group(0))
        return "[" + ", ".join(f"[{n}]({refs[n]})" for n in nums) + "]"

    out, pos = [], 0
    for s in SKIP_RE.finditer(body):  # leave code, math and comments untouched
        out.append(CITE_RE.sub(cite, body[pos:s.start()]))
        out.append(s.group(0))
        pos = s.end()
    out.append(CITE_RE.sub(cite, body[pos:]))
    return "".join(out)


def process(text: str, found: list | None = None) -> str:
    refs = references(text)
    m = re.search(r"^## Learn more\s*$", text, re.MULTILINE)
    if not refs or not m:
        return text
    return link_body(text[:m.start()], refs, found) + text[m.start():]


def main(argv: list[str]) -> int:
    files = sorted((ROOT / "lessons").glob("*.md"))
    bad = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        found: list[str] = []
        new = process(text, found)
        if "--list" in argv:
            refs = references(text)
            for c in found:
                print(f"{f.name}: {c} -> {', '.join(refs[n.strip()] for n in c.strip('[]').split(','))}")
        if new != text:
            bad.append(f.name)
            if "--check" not in argv and "--list" not in argv:
                f.write_text(new, encoding="utf-8")
                print("linked:", f.name)
    if "--check" in argv and bad:
        print("unlinked citations in:", ", ".join(bad), "(run scripts/link_citations.py)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
