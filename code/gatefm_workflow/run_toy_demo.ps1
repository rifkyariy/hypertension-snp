param(
  [string]$Python = 'python',
  [string]$Name = ('toy_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $root ('outputs\' + $Name)
if (Test-Path -LiteralPath $out) { throw "Output exists; choose a fresh -Name: $out" }
Write-Host "[GateFM] Run 1200x30 toy, 3 short Stage 1 fits, fixed-pattern Stage 2"
& $Python (Join-Path $root 'run_workflow.py') smoke --output $out --execute
if ($LASTEXITCODE -ne 0) { throw "Smoke pipeline failed; examine $out\run.log" }
Write-Host "COMPLETED $out"
Write-Host "FIGURES $out\figures"
Write-Host "DATA $out\toy_summary.json"
