param([ValidateRange(1,61)][int]$Workers = 8,
      [ValidateSet('normal','below-normal')][string]$Priority = 'below-normal')
$ErrorActionPreference = 'Stop'
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$TaskPython = Join-Path $TaskRoot 'experiment/.venv/Scripts/python.exe'
$TaskScript = Join-Path $TaskRoot 'experiment/scripts/analysis/run_scoped_production.py'
$TaskParent = Join-Path $TaskRoot 'experiment/data/08_reporting/research_v2'
$TaskPlan = @(
    @{ Run='bahia145_cpu16_2026-10-15'; Dataset='BA'; Control=$true },
    @{ Run='m5_zabraoui2408_cpu16_2026-10-15'; Dataset='M5'; Control=$true }
)
$TaskChain = Join-Path $TaskParent 'scoped_execution_2026-10-15.json'
$TaskState = @{ state='running'; pid=$PID; started_local=(Get-Date -Format o); workers_total=$Workers; priority=$Priority; order=@($TaskPlan | ForEach-Object { $_.Run }); national='Cancelled by user; removed from automatic execution.' }
$TaskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TaskChain -Encoding utf8
foreach ($TaskExperiment in $TaskPlan) {
    $TaskRun = $TaskExperiment.Run
    $TaskOutput = Join-Path $TaskParent $TaskRun
    $TaskStatusPath = Join-Path $TaskOutput 'status.json'
    if (Test-Path -LiteralPath $TaskStatusPath) {
        $TaskStatus = Get-Content -LiteralPath $TaskStatusPath -Raw | ConvertFrom-Json
        if ($TaskStatus.state -eq 'completed') { continue }
        if ($TaskStatus.state -eq 'cancelled_by_user') { continue }
        $TaskExisting = Get-CimInstance Win32_Process -Filter "ProcessId=$($TaskStatus.pid)"
        if ($TaskExisting -and $TaskExisting.CommandLine -like "*run_scoped_production.py*--run $TaskRun*") {
            throw "Fila ja ativa: $TaskRun, PID $($TaskStatus.pid)"
        }
    }
    $TaskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $TaskStdout = Join-Path $TaskOutput "stdout_$TaskStamp.log"
    $TaskStderr = Join-Path $TaskOutput "stderr_$TaskStamp.log"
    $TaskArgs = @('-u',('"'+$TaskScript+'"'),'--run',$TaskRun,'--dataset',$TaskExperiment.Dataset,
                  '--workers',$Workers,'--priority',$Priority,'--resume-execution-profile',
                  '--source','overnight_cpu16_2026-10-14','--source','census_2026-10-14')
    if ($TaskExperiment.Control) { $TaskArgs += '--include-dqn' }
    if ($TaskExperiment.Dataset -eq 'BR') { $TaskArgs += @('--reuse-completed-bahia','bahia145_cpu16_2026-10-15') }
    $TaskState['active_run']=$TaskRun
    $TaskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TaskChain -Encoding utf8
    $TaskProcess = Start-Process -FilePath $TaskPython -ArgumentList $TaskArgs -WorkingDirectory $TaskRoot -WindowStyle Hidden -RedirectStandardOutput $TaskStdout -RedirectStandardError $TaskStderr -PassThru -Wait
    if ($TaskProcess.ExitCode -ne 0) {
        $TaskState['state']='failed';$TaskState['exit_code']=$TaskProcess.ExitCode
        $TaskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TaskChain -Encoding utf8
        throw "Falha em $TaskRun; consultar $TaskStderr"
    }
}
$TaskState['state']='completed';$TaskState['finished_local']=(Get-Date -Format o)
$TaskState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $TaskChain -Encoding utf8
