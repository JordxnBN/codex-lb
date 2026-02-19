param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 2455,
  [int]$HealthTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
  throw "Python not found at '$pythonPath'. Run setup first (.venv creation + deps install)."
}

$existing = Get-NetTCPConnection -LocalAddress $HostAddress -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1

if ($null -ne $existing) {
  Write-Host "codex-lb already running on $HostAddress`:$Port (PID $($existing.OwningProcess))."
  exit 0
}

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $HostAddress, "--port", "$Port")

$process = Start-Process -FilePath $pythonPath -ArgumentList $uvicornArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
Write-Host "Started codex-lb (PID $($process.Id)). Waiting for health check..."

$deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
$healthOk = $false

while ((Get-Date) -lt $deadline) {
  try {
    $response = Invoke-WebRequest -Uri "http://$HostAddress`:$Port/health" -UseBasicParsing -TimeoutSec 3
    if ($response.StatusCode -eq 200) {
      $healthOk = $true
      break
    }
  }
  catch {
    Start-Sleep -Milliseconds 500
  }
}

if (-not $healthOk) {
  throw "codex-lb did not become healthy within $HealthTimeoutSeconds seconds."
}

Write-Host "codex-lb is up at http://$HostAddress`:$Port"
