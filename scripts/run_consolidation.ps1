param(
    [switch]$DryRun,
    [int]$MaxIterations = 5,
    [string]$RunDir = "",
    [switch]$InstallDeps,
    [int]$RequiredConsecutivePasses = 0,
    [string]$LlmRoutingConfig = ""
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

function Get-RoleConfigValue {
    param(
        $ConfigObject,
        [string[]]$PathParts,
        $DefaultValue = $null
    )

    $current = $ConfigObject
    foreach ($part in $PathParts) {
        if ($null -eq $current) {
            return $DefaultValue
        }
        if ($current -is [System.Collections.IDictionary]) {
            if (-not $current.Contains($part)) {
                return $DefaultValue
            }
            $current = $current[$part]
            continue
        }
        $property = $current.PSObject.Properties[$part]
        if ($null -eq $property) {
            return $DefaultValue
        }
        $current = $property.Value
    }
    if ($null -eq $current -or $current -eq "") {
        return $DefaultValue
    }
    return $current
}

function Get-AgentExecutableDefault {
    param([string]$Client)

    switch ($Client) {
        "claude" { return "claude" }
        "opencode" { return "opencode" }
        default { return "codex" }
    }
}

function Assert-CommandAvailable {
    param([string]$Executable)

    if (-not (Get-Command $Executable -ErrorAction SilentlyContinue)) {
        throw "CLI '$Executable' is not available on PATH."
    }
}

function Assert-EnvPresent {
    param([string]$EnvVarName)

    if (-not (Get-Item -Path "Env:$EnvVarName" -ErrorAction SilentlyContinue)) {
        throw "$EnvVarName is not set."
    }
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

$routingConfig = $null
if ($LlmRoutingConfig) {
    if (-not (Test-Path $LlmRoutingConfig)) {
        throw "LLM routing config not found: $LlmRoutingConfig"
    }
    $routingConfig = Get-Content $LlmRoutingConfig -Raw | ConvertFrom-Json
}

$promptRoles = @(
    @("single_prompt", "tier2_primary", "client", "genai"),
    @("single_prompt", "tier2_secondary", "client", "genai"),
    @("single_prompt", "taxonomy_primary", "client", "genai"),
    @("single_prompt", "taxonomy_secondary", "client", "genai"),
    @("single_prompt", "tier3_judge_primary", "client", "genai"),
    @("single_prompt", "tier3_judge_secondary", "client", "genai"),
    @("embeddings", "tier3", "client", "genai")
)

$requiredEnvVars = New-Object System.Collections.Generic.HashSet[string]
foreach ($role in $promptRoles) {
    $client = [string](Get-RoleConfigValue $routingConfig @($role[0], $role[1], $role[2]) $role[3]).ToLowerInvariant()
    switch ($client) {
        "genai" { [void]$requiredEnvVars.Add("GOOGLE_API_KEY") }
        "openai" { [void]$requiredEnvVars.Add("OPENAI_API_KEY") }
        "openrouter" { [void]$requiredEnvVars.Add("OPENROUTER_API_KEY") }
    }
}
foreach ($envVar in $requiredEnvVars) {
    Assert-EnvPresent $envVar
}

$reviewClient = [string](Get-RoleConfigValue $routingConfig @("agents", "review", "client") "codex").ToLowerInvariant()
$reviewExecutable = [string](Get-RoleConfigValue $routingConfig @("agents", "review", "executable") (Get-AgentExecutableDefault $reviewClient))
$tailClient = [string](Get-RoleConfigValue $routingConfig @("agents", "taxonomy_tail", "client") "codex").ToLowerInvariant()
$tailExecutable = [string](Get-RoleConfigValue $routingConfig @("agents", "taxonomy_tail", "executable") (Get-AgentExecutableDefault $tailClient))
Assert-CommandAvailable $reviewExecutable
Assert-CommandAvailable $tailExecutable

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

if ($LlmRoutingConfig) {
    $args += @("--llm-routing-config", $LlmRoutingConfig)
}

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
