param(
    [int]$ExistingBuildPid = 0,
    [int]$MaxRetryPasses = 5,
    [int]$Parallel = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourcesDir = Join-Path (Split-Path -Parent $repoRoot) "quant-rnd-export\sources"
$runsDir = Join-Path $repoRoot "runs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $runsDir "quant_rnd_pipeline_supervisor_$timestamp.log"

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-GraphStatus {
    Set-Location $repoRoot
    $pythonScript = @"
import json
from notebooklm_graph_pipe.runtime.graph_builder_runtime import GraphBuilderAPI

api = GraphBuilderAPI(
    neo4j_uri="bolt://host.docker.internal:7687",
    neo4j_user="neo4j",
    neo4j_password="password123",
    neo4j_database="neo4j",
)
print(json.dumps(api.sources_list()))
"@
    $responseJson = $pythonScript | python -
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query graph status through notebooklm_graph_pipe.runtime.graph_builder_runtime"
    }
    $response = @()
    if ($responseJson) {
        $response = @($responseJson | ConvertFrom-Json)
    }

    $statusByName = @{}
    foreach ($item in $response) {
        if ($item.fileName) {
            $statusByName[$item.fileName] = $item.status
        }
    }

    $localAll = @(Get-ChildItem -Path $sourcesDir -Filter *.txt | Select-Object -ExpandProperty Name)
    $localGe10 = @(Get-ChildItem -Path $sourcesDir -Filter *.txt | Where-Object { $_.Length -ge 10 } | Select-Object -ExpandProperty Name)

    $completedGe10 = @($localGe10 | Where-Object { $statusByName[$_] -eq "Completed" })
    $remainingGe10 = @($localGe10 | Where-Object { $statusByName[$_] -ne "Completed" })
    $remainingAll = @($localAll | Where-Object { $statusByName[$_] -ne "Completed" })

    [pscustomobject]@{
        local_total = $localAll.Count
        local_ge10 = $localGe10.Count
        db_total = $statusByName.Count
        completed_ge10 = $completedGe10.Count
        remaining_ge10 = $remainingGe10.Count
        remaining_all = $remainingAll.Count
        remaining_sample = @($remainingAll | Select-Object -First 10)
    }
}

function Invoke-BuildPass {
    param(
        [string]$Label,
        [int]$MinFileSize,
        [switch]$SkipPostprocess,
        [switch]$SkipUpload,
        [switch]$SkipExtract
    )

    Set-Location $repoRoot
    $passLog = Join-Path $runsDir "$Label.log"
    $args = @(
        "-u",
        "-m", "notebooklm_graph_pipe.cli.build_graph",
        "--neo4j-uri", "bolt://host.docker.internal:7687",
        "--neo4j-user", "neo4j",
        "--neo4j-password", "password123",
        "--neo4j-database", "neo4j",
        "--model", "google_flash",
        "--sources-dir", "..\quant-rnd-export\sources",
        "--parallel", [string]$Parallel,
        "--min-file-size", [string]$MinFileSize
    )

    if ($SkipPostprocess) {
        $args += "--skip-postprocess"
    }
    if ($SkipUpload) {
        $args += "--skip-upload"
    }
    if ($SkipExtract) {
        $args += "--skip-extract"
    }

    Write-Log "starting build pass label=$Label"
    Write-Log "command=python $($args -join ' ')"
    & python @args 2>&1 | Tee-Object -FilePath $passLog -Append
    $exitCode = $LASTEXITCODE
    Write-Log "finished build pass label=$Label exit_code=$exitCode log=$passLog"
    return $exitCode
}

New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
Write-Log "supervisor started repo_root=$repoRoot"
Write-Log "config parallel=$Parallel max_retry_passes=$MaxRetryPasses"

if ($ExistingBuildPid -gt 0) {
    Write-Log "waiting for existing build_graph pid=$ExistingBuildPid"
    while (Get-Process -Id $ExistingBuildPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Write-Log "existing build_graph process exited"
}

for ($attempt = 1; $attempt -le $MaxRetryPasses; $attempt++) {
    $status = Get-GraphStatus
    Write-Log ("status_before_retry_{0}={1}" -f $attempt, ($status | ConvertTo-Json -Compress -Depth 4))
    if ([int]$status.remaining_ge10 -le 0) {
        break
    }

    $retryLabel = "quant_rnd_retry_ge10_{0}_{1}" -f $attempt, (Get-Date -Format "yyyyMMdd_HHmmss")
    Invoke-BuildPass -Label $retryLabel -MinFileSize 10 -SkipPostprocess | Out-Null
}

$statusAfterGe10 = Get-GraphStatus
Write-Log ("status_after_ge10={0}" -f ($statusAfterGe10 | ConvertTo-Json -Compress -Depth 4))

if ([int]$statusAfterGe10.remaining_all -gt 0) {
    $allLabel = "quant_rnd_retry_all_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    Invoke-BuildPass -Label $allLabel -MinFileSize 1 -SkipPostprocess | Out-Null
}

$statusBeforePostprocess = Get-GraphStatus
Write-Log ("status_before_postprocess={0}" -f ($statusBeforePostprocess | ConvertTo-Json -Compress -Depth 4))

if ([int]$statusBeforePostprocess.remaining_ge10 -eq 0) {
    $postLabel = "quant_rnd_postprocess_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
    $postCode = Invoke-BuildPass -Label $postLabel -MinFileSize 1 -SkipUpload -SkipExtract
    if ($postCode -eq 0) {
        $consolidationLog = Join-Path $runsDir ("quant_rnd_consolidation_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        Write-Log "starting consolidation"
        Set-Location $repoRoot
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\run_consolidation.ps1" -InstallDeps 2>&1 |
            Tee-Object -FilePath $consolidationLog -Append
        Write-Log "finished consolidation exit_code=$LASTEXITCODE log=$consolidationLog"
    } else {
        Write-Log "postprocess-only pass failed; consolidation skipped"
    }
} else {
    Write-Log "non-tiny sources still incomplete after retries; consolidation skipped"
}

$finalStatus = Get-GraphStatus
Write-Log ("final_status={0}" -f ($finalStatus | ConvertTo-Json -Compress -Depth 4))
Write-Log "supervisor finished"
