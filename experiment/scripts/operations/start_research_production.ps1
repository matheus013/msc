param([string]$Run = 'portfolio6_reviewed_2026-10-14',
      [ValidateRange(1,64)][int]$Workers = 8,
      [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
      [ValidateSet('normal','below-normal')][string]$Priority = 'below-normal')
$ErrorActionPreference = 'Stop'
if ($Run -notmatch '^[A-Za-z0-9_-]+$') { throw 'Nome de rodada invalido.' }
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$TaskPython = Join-Path $TaskRoot 'experiment/.venv/Scripts/python.exe'
$TaskScript = Join-Path $TaskRoot 'experiment/scripts/analysis/run_production_v2.py'
$TaskOutput = Join-Path $TaskRoot "experiment/data/08_reporting/research_v2/$Run"
New-Item -ItemType Directory -Force -Path $TaskOutput | Out-Null
$TaskStatus = Join-Path $TaskOutput 'status.json'
if (Test-Path -LiteralPath $TaskStatus) {
    $TaskCurrent = Get-Content -LiteralPath $TaskStatus -Raw | ConvertFrom-Json
    if ($TaskCurrent.state -eq 'completed') { Write-Output 'Rodada ja concluida.'; exit 0 }
    $TaskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($TaskCurrent.pid)"
    if ($TaskProcess -and $TaskProcess.CommandLine -like "*run_production_v2.py*--run $Run*") {
        Write-Output "Fila ja ativa: PID $($TaskCurrent.pid)"; exit 0
    }
}
$TaskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$TaskStdout = Join-Path $TaskOutput "stdout_$TaskStamp.log"
$TaskStderr = Join-Path $TaskOutput "stderr_$TaskStamp.log"
$TaskArgs = @('-u', ('"' + $TaskScript + '"'), '--run', $Run, '--workers', $Workers, '--device', $Device, '--priority', $Priority)
$TaskChild = Start-Process -FilePath $TaskPython -ArgumentList $TaskArgs -WorkingDirectory $TaskRoot -WindowStyle Hidden -RedirectStandardOutput $TaskStdout -RedirectStandardError $TaskStderr -PassThru
$TaskLaunch = @{ pid=$TaskChild.Id; started_local=(Get-Date -Format o); stdout=$TaskStdout; stderr=$TaskStderr; command="python run_production_v2.py --run $Run" }
$TaskLaunch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $TaskOutput 'launcher.json') -Encoding utf8
Write-Output "Fila iniciada: PID $($TaskChild.Id)"
Write-Output "Resultados: $TaskOutput"
