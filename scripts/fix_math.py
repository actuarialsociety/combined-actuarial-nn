#!/usr/bin/env python3
r"""Make LaTeX in Markdown safe for GitHub (repo view and wiki) as well as Marp/KaTeX.

GitHub applies Markdown backslash escapes before rendering math, so `\{` becomes `{`
(breaking `\big\{` with "Missing or unrecognized delimiter for \big"), `\;` becomes a
literal `;` and `\\` a single backslash. Inside $...$ and $$...$$ this rewrites them to
escape-free macros that render the same everywhere:

    \{ \}  ->  \lbrace \rbrace        \, \; \!  ->  \thinspace \medspace \negthinspace
    \|     ->  \Vert                  \\        ->  \cr

Display math is written as ```math fences rather than $$ blocks: inside <details>
panels GitHub renders $$ blocks unreliably. marp.config.mjs converts the fences
back to $$ for the slides.

    python3 scripts/fix_math.py              # README.md and lessons/*.md, in place
    python3 scripts/fix_math.py --check      # exit 1 if anything needs fixing (CI)
    python3 scripts/fix_math.py --github     # also render via GitHub's API and confirm
                                             # every math span became math (GITHUB_TOKEN)
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
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


# In inline math, `_` right after punctuation (`}_{i,j}`, `)_t`) can open Markdown
# emphasis on GitHub and stop the span rendering as math; `}_ {` means the same in TeX.
EMPH_RE = re.compile(r"(?<=[^A-Za-z0-9\s\\])_(?=\S)")


def fix_math(span: str) -> str:
    # A space is needed after the new macro name only when a letter follows it.
    span = ESC_RE.sub(lambda m: SUBS[m.group(1)] + (" " + m.group(2) if m.group(2) else ""), span)
    if span.startswith("$") and not span.startswith("$$"):
        span = EMPH_RE.sub("_ ", span)
    return span


DISPLAY_RE = re.compile(r"^\$\$[ \t]*\n(.*?)\n\$\$[ \t]*$", re.MULTILINE | re.DOTALL)


def display_to_fence(text: str) -> str:
    """$$ blocks -> ```math fences. Inside <details> panels GitHub renders $$ blocks
    unreliably (later ones show as raw text); ```math fences render everywhere, and
    marp.config.mjs turns them back into $$ for the slides."""
    return DISPLAY_RE.sub(lambda m: f"```math\n{m.group(1)}\n```", text)


DETAILS_RE = re.compile(r"<details>.*?</details>", re.DOTALL)
FENCE_RE = re.compile(r"^```math[ \t]*\n(.*?)\n```[ \t]*$", re.MULTILINE | re.DOTALL)


def panel_display_inline(text: str) -> str:
    """Inside <details> panels GitHub does not render display math in a panel that also
    has inline math, so display equations there become one-line $\\displaystyle ...$."""

    def one_line(m: re.Match) -> str:
        body = " ".join(line.strip() for line in m.group(1).splitlines() if line.strip())
        return f"$\\displaystyle {body}$"

    return DETAILS_RE.sub(lambda d: FENCE_RE.sub(one_line, d.group(0)), text)


def fix_text(text: str) -> str:
    """Rewrite math spans and ```math fences, leaving other code blocks untouched."""
    text = panel_display_inline(display_to_fence(text))
    parts = re.split(r"(^```.*?^```\n?)", text, flags=re.MULTILINE | re.DOTALL)
    return "".join(
        (fix_math(p) if p.startswith("```math") else p)
        if p.startswith("```")
        else MATH_RE.sub(lambda m: fix_math(m.group(0)), p)
        for p in parts
    )


def expected_math(text: str) -> tuple[int, int]:
    """Number of display (```math) and inline ($...$) math spans in a file."""
    parts = re.split(r"(^```.*?^```\n?)", text, flags=re.MULTILINE | re.DOTALL)
    display = sum(p.startswith("```math") for p in parts)
    inline = sum(len(MATH_RE.findall(p)) for p in parts if not p.startswith("```"))
    return display, inline


def github_rendered(text: str, token: str) -> tuple[int, int]:
    req = urllib.request.Request(
        "https://api.github.com/markdown",
        data=json.dumps({"text": text, "mode": "gfm"}).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode()
    return html.count("js-display-math"), html.count("js-inline-math")


def check_github(files: list[Path]) -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("--github needs GITHUB_TOKEN (or GH_TOKEN)")
        return 1
    bad = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        exp, got = expected_math(text), github_rendered(text, token)
        ok = exp == got
        bad += not ok
        counts = f"display {got[0]}/{exp[0]}, inline {got[1]}/{exp[1]}"
        print(f"{'ok  ' if ok else 'FAIL'} {f.relative_to(ROOT)}: {counts}")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    check = "--check" in argv or "--github" in argv
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
    if check and changed:
        return 1
    return check_github(files) if "--github" in argv else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
