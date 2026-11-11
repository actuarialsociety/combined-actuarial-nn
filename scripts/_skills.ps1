# Shared helpers for the update_*_skills.ps1 scripts (dot-source: . "$PSScriptRoot/_skills.ps1").
# PowerShell twin of the logic repeated in the update_*_skills.sh scripts: download a GitHub
# repository as a tarball, then copy one skill folder and the repository's LICENSE into
# .claude/skills/. Needs PowerShell 7 and tar (built into Windows 10+ and macOS).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Get-Repo([string]$Repo, [string]$Ref = "main") {
    # Download github.com/$Repo at $Ref and return the folder it was extracted to.
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("skills-" + [guid]::NewGuid())
    New-Item -ItemType Directory $dir | Out-Null
    $tgz = Join-Path $dir "repo.tar.gz"
    Invoke-WebRequest "https://codeload.github.com/$Repo/tar.gz/$Ref" -OutFile $tgz
    tar -xzf $tgz -C $dir --strip-components=1
    Remove-Item $tgz
    return $dir
}

function Copy-Skill([string]$RepoDir, [string]$SkillPath, [string]$Name = "") {
    # Replace .claude/skills/<name> with $RepoDir/$SkillPath plus the repository LICENSE.
    if (-not $Name) { $Name = Split-Path $SkillPath -Leaf }
    $dest = ".claude/skills/$Name"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Copy-Item -Recurse (Join-Path $RepoDir $SkillPath) $dest
    Copy-Item (Join-Path $RepoDir "LICENSE") "$dest/LICENSE"
    Write-Host "vendored $Name"
}

function Copy-SkillFile([string]$RepoDir, [string]$Name) {
    # Only skills/<name>/SKILL.md and the LICENSE, as the exa and kedro bash scripts do.
    $dest = ".claude/skills/$Name"
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $RepoDir "skills/$Name/SKILL.md") "$dest/SKILL.md"
    Copy-Item (Join-Path $RepoDir "LICENSE") "$dest/LICENSE"
    Write-Host "vendored $Name"
}
