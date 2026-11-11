#!/usr/bin/env python3
"""Every folder's README.md must describe the code that is really there (AGENTS.md §7).

    python3 scripts/check_readmes.py      # exit 1 and list each problem; run by the pre-push
                                          # hook (scripts/install_hooks.sh) and by CI

For each folder that holds tracked files (vendored skills, data and tool state excluded; the
project README, which is the wiki's home page, only needs its diagram):

1. `README.md` exists and has a ```` ```mermaid ```` diagram.
2. Every code file in the folder (.py, .sh, .ps1) appears in the diagram, by file name or
   stem: a new module cannot be added without drawing it.
3. Every other file is named in the README, directly or by a backticked glob (`*.png`).
4. Every backticked file name in the README (`plots.py`) exists somewhere in the repo:
   a deleted or renamed file cannot linger in the docs.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SKIP = (".claude/", ".agents/", ".viz/", ".omc/", "data/", ".devcontainer/")  # not our code
IGNORED_FILES = {"README.md", ".gitkeep", ".gitignore"}
CODE = {".py", ".sh", ".ps1"}
MERMAID = re.compile(r"```mermaid\n(.*?)```", re.S)
TICKED = re.compile(r"`([^`\s]+)`")
FILE_LIKE = re.compile(r"^[\w.*-]+\.(py|sh|ps1|ya?ml|ipynb|json|csv|toml|png)$")


def tracked() -> list[PurePosixPath]:
    """Tracked files plus untracked, not-ignored ones (so a new file is checked before commit)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [PurePosixPath(p) for p in out.splitlines() if not p.startswith(SKIP)]


def problems() -> list[str]:
    files = tracked()
    names = {p.name for p in files}
    folders: dict[PurePosixPath, list[str]] = defaultdict(list)
    for p in files:
        folders[p.parent].append(p.name)
    out = []
    for folder, here in sorted(folders.items()):
        readme = ROOT / folder / "README.md"
        where = f"{folder}/README.md" if str(folder) != "." else "README.md"
        if not readme.exists():
            out.append(f"{where}: missing")
            continue
        text = readme.read_text(encoding="utf-8")
        diagram = "\n".join(MERMAID.findall(text))
        if not diagram:
            out.append(f"{where}: no ```mermaid diagram")
        ticked = set(TICKED.findall(text))
        globs = [t for t in ticked if "*" in t]
        if str(folder) == ".":  # the project README is the wiki home: diagram only
            continue
        for name in sorted(set(here) - IGNORED_FILES):
            suffix = PurePosixPath(name).suffix
            stem = PurePosixPath(name).stem
            if suffix in CODE and name not in diagram and stem not in diagram:
                out.append(f"{where}: `{name}` is not in the diagram")
            if name not in text and not any(fnmatch.fnmatch(name, g) for g in globs):
                out.append(f"{where}: `{name}` is not described")
        for t in sorted(ticked):
            base = t.rsplit("/", 1)[-1]
            if FILE_LIKE.match(base) and "*" not in base and base not in names:
                out.append(f"{where}: `{t}` does not exist")
    return out


def main() -> int:
    found = problems()
    for p in found:
        print(p)
    print(f"{len(found)} README problem(s)" if found else "READMEs match the code")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
