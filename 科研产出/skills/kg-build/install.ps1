param(
    [string]$Target = (Join-Path $HOME ".agents\skills\kg-build"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$src = $PSScriptRoot

if (-not (Test-Path (Join-Path $src "SKILL.md"))) {
    throw "源目录不是 kg-build skill：$src"
}

if (Test-Path $Target) {
    if (-not $Force) {
        throw "目标已存在：$Target（用 -Force 覆盖）"
    }
    Remove-Item -Recurse -Force $Target
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $src "*") $Target

if (-not (Test-Path (Join-Path $Target "SKILL.md"))) {
    throw "安装失败：目标目录没有 SKILL.md"
}

Write-Host "installed: $Target"
Write-Host "next: open a new session, then the 'kg-build' skill should appear in the skill catalog."
