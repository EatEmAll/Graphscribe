param(
    [switch]$DryRun,
    [int]$MaxIterations = 5,
    [string]$RunDir = "",
    [switch]$InstallDeps,
    [int]$RequiredConsecutivePasses = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Set-Neo4jEnvFromCodexConfig {
    param([string]$ConfigPath)

    if (-not (Test-Path $ConfigPath)) {
        return
    }

    $content = Get-Content $ConfigPath
    $inNeo4jEnv = $false
    foreach ($line in $content) {
        if ($line -match '^\[mcp_servers\.neo4j\.env\]') {
            $inNeo4jEnv = $true
            continue
        }
        if ($inNeo4jEnv -and $line -match '^\[') {
            break
        }
        if (-not $inNeo4jEnv) {
            continue
        }
        if ($line -match '^\s*NEO4J_URI\s*=\s*"([^"]+)"') {
            if (-not $env:NEO4J_URI) { $env:NEO4J_URI = $Matches[1] }
        } elseif ($line -match '^\s*NEO4J_USERNAME\s*=\s*"([^"]+)"') {
            if (-not $env:NEO4J_USERNAME) { $env:NEO4J_USERNAME = $Matches[1] }
        } elseif ($line -match '^\s*NEO4J_PASSWORD\s*=\s*"([^"]+)"') {
            if (-not $env:NEO4J_PASSWORD) { $env:NEO4J_PASSWORD = $Matches[1] }
        } elseif ($line -match '^\s*NEO4J_DATABASE\s*=\s*"([^"]+)"') {
            if (-not $env:NEO4J_DATABASE) { $env:NEO4J_DATABASE = $Matches[1] }
        }
    }
}

function Ensure-Dependencies {
    python -c "import google.genai, neo4j, numpy" *> $null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "Installing Python dependencies..."
    python -m pip install -r (Join-Path $vendorRepoRoot "backend\requirements.txt") -c (Join-Path $vendorRepoRoot "backend\constraints.txt")
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$vendorRepoRoot = Join-Path $repoRoot "vendor\llm-graph-builder"
Set-Location $repoRoot

Write-Host "[run_consolidation] repo_root=$repoRoot"

Set-Neo4jEnvFromCodexConfig -ConfigPath (Join-Path $repoRoot ".codex\config.toml")

if ($InstallDeps) {
    Write-Host "[run_consolidation] checking dependencies"
    Ensure-Dependencies
}

if (-not $env:GOOGLE_API_KEY) {
    throw "GOOGLE_API_KEY is not set."
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex CLI is not available on PATH."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $RunDir) {
    $RunDir = if ($DryRun) {
        "runs/consolidation_dry_codex_$timestamp"
    } else {
        "runs/consolidation_full_codex_$timestamp"
    }
}

$effectiveRequiredConsecutivePasses = $RequiredConsecutivePasses
if ($effectiveRequiredConsecutivePasses -le 0) {
    $effectiveRequiredConsecutivePasses = if ($DryRun) { 2 } else { 1 }
}

$args = @(
    "consolidate_self_improving.py",
    "--max-iterations", $MaxIterations,
    "--target-concept-ratio", "0.05",
    "--target-duplicate-rate", "0.015",
    "--target-concept-without-taxonomy-ratio", "0.60",
    "--required-consecutive-passes", $effectiveRequiredConsecutivePasses,
    "--taxonomy-plateau-reviews", "2",
    "--taxonomy-plateau-min-delta", "0.002",
    "--run-dir", $RunDir
)

if ($DryRun) {
    $args += "--dry-run"
}

$mode = "full-run"
if ($DryRun) {
    $mode = "dry-run"
}
Write-Host "[run_consolidation] mode=$mode"
Write-Host "[run_consolidation] run_dir=$RunDir"
Write-Host "[run_consolidation] required_consecutive_passes=$effectiveRequiredConsecutivePasses"
Write-Host "[run_consolidation] command=python -u $($args -join ' ')"

$env:PYTHONUNBUFFERED = "1"
python -u @args
