# One-click version bump & release script
# Usage:
#   .\update_version.ps1                 # auto bump patch (0.1.0 -> 0.1.1)
#   .\update_version.ps1 -Part minor     # bump minor  (0.1.0 -> 0.2.0)
#   .\update_version.ps1 -Part major     # bump major  (0.1.0 -> 1.0.0)
#   .\update_version.ps1 -Version 2.3.4  # set explicit version
#   .\update_version.ps1 -NoPush         # only bump + commit + tag, no push
#
# Pushing a v* tag triggers .github/workflows/build-release.yml,
# which builds cpu/cuda packages and publishes a GitHub Release.

param(
    [ValidateSet("major", "minor", "patch")]
    [string]$Part = "patch",
    [string]$Version,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$initFile = Join-Path $PSScriptRoot "src\__init__.py"
if (-not (Test-Path $initFile)) {
    Write-Host "ERROR: src\__init__.py not found." -ForegroundColor Red
    exit 1
}

# --- read current version -------------------------------------------------
$content = Get-Content $initFile -Raw -Encoding UTF8
$match = [regex]::Match($content, '__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"')
if (-not $match.Success) {
    Write-Host "ERROR: __version__ not found in src\__init__.py." -ForegroundColor Red
    exit 1
}
$cur = "{0}.{1}.{2}" -f $match.Groups[1].Value, $match.Groups[2].Value, $match.Groups[3].Value
Write-Host "Current version: v$cur" -ForegroundColor Cyan

# --- compute new version --------------------------------------------------
if ($Version) {
    if ($Version -notmatch '^\d+\.\d+\.\d+$') {
        Write-Host "ERROR: -Version must look like 1.2.3" -ForegroundColor Red
        exit 1
    }
    $new = $Version
} else {
    $v = @([int]$match.Groups[1].Value, [int]$match.Groups[2].Value, [int]$match.Groups[3].Value)
    switch ($Part) {
        "major" { $v[0]++; $v[1] = 0; $v[2] = 0 }
        "minor" { $v[1]++; $v[2] = 0 }
        "patch" { $v[2]++ }
    }
    $new = $v -join "."
}

if ($new -eq $cur) {
    Write-Host "ERROR: new version v$new equals current version, abort." -ForegroundColor Red
    exit 1
}
Write-Host "New version:     v$new ($Part bump)" -ForegroundColor Green

# --- write back (UTF-8 without BOM, PS5.1 Set-Content would add BOM) -------
$content = $content -replace '(?s)(__version__\s*=\s*")\d+\.\d+\.\d+(")', ('$1' + $new + '$2')
[System.IO.File]::WriteAllText($initFile, $content, (New-Object System.Text.UTF8Encoding($false)))

# --- git commit + tag ------------------------------------------------------
$tag = "v$new"

$dirty = git status --porcelain -- src/__init__.py
if ($dirty) {
    git add src/__init__.py
    git commit -m "chore: release $tag"
    Write-Host "Committed: chore: release $tag" -ForegroundColor Green
} else {
    Write-Host "src/__init__.py has no diff to commit (version file unchanged?)." -ForegroundColor Yellow
}

git tag -a $tag -m "Release $tag"
Write-Host "Tagged: $tag" -ForegroundColor Green

# --- push (triggers CI release build) --------------------------------------
if ($NoPush) {
    Write-Host "-NoPush specified. Run later:  git push origin main --follow-tags" -ForegroundColor Yellow
    exit 0
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
git push origin $branch
git push origin $tag
Write-Host ""
Write-Host "Pushed $branch and $tag. GitHub Actions 'Build Windows Release' is now building" -ForegroundColor Green
Write-Host "cpu/cuda packages and will publish Release $tag automatically." -ForegroundColor Green
