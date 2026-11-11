# `.githooks/`

Git hooks kept in the repo. Enable once per clone with `scripts/install_hooks.sh` (or
`scripts/install_hooks.ps1` on Windows), which sets `core.hooksPath`.

```mermaid
flowchart LR
    push[git push] --> hook["pre-push"]
    hook --> readmes["check_readmes.py<br/>READMEs match the code"]
    hook --> changelog["CHANGELOG.md changed<br/>vs origin/develop"]
    readmes & changelog -->|both pass| remote[(GitHub)]
```

| Hook | Checks |
| --- | --- |
| `pre-push` | every folder README matches its code; the branch adds a CHANGELOG.md line |
