param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$TaskPython = Join-Path $TaskRoot 'experiment/.venv/Scripts/python.exe'
$TaskScript = Join-Path $TaskRoot 'experiment/scripts/analysis/run_aipe_pipeline.py'
$TaskRuns = @(
    @{ Dataset='BA'; Run='aipe_ba145_unified_2026-10-15' },
    @{ Dataset='M5'; Run='aipe_m5_2408_unified_2026-10-15' }
)
foreach ($TaskRun in $TaskRuns) {
    $TaskFolder = Join-Path $TaskRoot ('experiment/data/08_reporting/research_v2/' + $TaskRun.Run)
    if (-not $Execute) {
        & $TaskPython $TaskScript plan --dataset $TaskRun.Dataset --run $TaskRun.Run
        if ($LASTEXITCODE -ne 0) { throw "Falha ao planejar $($TaskRun.Run)" }
        continue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $TaskFolder 'plan.json'))) { throw 'Preparar e relatar o plano antes de executar.' }
    $TaskArguments = @('-u',('"' + $TaskScript + '"'),'execute','--dataset',$TaskRun.Dataset,'--run',$TaskRun.Run)
    $TaskProcess = Start-Process -FilePath $TaskPython -ArgumentList $TaskArguments -WorkingDirectory $TaskRoot -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $TaskFolder 'runner.stdout.log') -RedirectStandardError (Join-Path $TaskFolder 'runner.stderr.log')
    if ($TaskProcess.ExitCode -ne 0) { throw "Falha em $($TaskRun.Run); consultar runner.stderr.log" }
}
