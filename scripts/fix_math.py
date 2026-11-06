#!/usr/bin/env python3
r"""Make LaTeX in Markdown safe for GitHub (repo view and wiki) as well as Marp/KaTeX.

GitHub applies Markdown backslash escapes before rendering math, so `\{` becomes `{`
(breaking `\big\{` with "Missing or unrecognized delimiter for \big"), `\;` becomes a
literal `;` and `\\` a single backslash. Inside $...$ and $$...$$ this rewrites them to
escape-free macros that render the same everywhere:

    \{ \}  ->  \lbrace \rbrace        \, \; \!  ->  \thinspace \medspace \negthinspace
    \|     ->  \Vert                  \\        ->  \cr

    python3 scripts/fix_math.py              # README.md and lessons/*.md, in place
    python3 scripts/fix_math.py --check      # exit 1 if anything needs fixing (CI)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATH_RE = re.compile(r"\$\$.+?\$\$|(?<![\\$])\$(?!\s)[^$\n]+?(?<![\s\\])\$(?!\$)", re.DOTALL)
SUBS = {
    "\\\\": r"\cr",
    r"\{": r"\lbrace",
    r"\}": r"\rbrace",
    r"\,": r"\thinspace",
    r"\;": r"\medspace",
    r"\!": r"\negthinspace",
    r"\|": r"\Vert",
}
ESC_RE = re.compile("(" + "|".join(re.escape(k) for k in SUBS) + r")([A-Za-z]?)")


def fix_math(span: str) -> str:
    # A space is needed after the new macro name only when a letter follows it.
    return ESC_RE.sub(lambda m: SUBS[m.group(1)] + (" " + m.group(2) if m.group(2) else ""), span)


def fix_text(text: str) -> str:
    """Rewrite math spans, leaving fenced code blocks untouched."""
    parts = re.split(r"(^```.*?^```\n?)", text, flags=re.MULTILINE | re.DOTALL)
    return "".join(
        p if p.startswith("```") else MATH_RE.sub(lambda m: fix_math(m.group(0)), p)
        for p in parts
    )


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files = [ROOT / "README.md", *sorted((ROOT / "lessons").glob("*.md"))]
    changed = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        new = fix_text(text)
        if new != text:
            changed.append(str(f.relative_to(ROOT)))
            if not check:
                f.write_text(new, encoding="utf-8")
    for c in changed:
        print(("needs fixing (run scripts/fix_math.py): " if check else "fixed: ") + c)
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
