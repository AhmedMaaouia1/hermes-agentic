Write-Host "=== FULL HERMES EVALUATION ==="

.\scripts\eval_categorization.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

.\scripts\eval_hierarchy.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

.\scripts\eval_reviewer.ps1
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "=== ALL EVALUATIONS COMPLETED SUCCESSFULLY ==="
