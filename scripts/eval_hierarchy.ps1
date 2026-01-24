Write-Host "=== Evaluating hierarchy ==="

docker compose run --rm hermes `
  python src/evaluation/evaluate.py `
  --ground-truth data/ground_truth.json `
  --predictions logs/planner/hierarchy.json `
  --output logs/planner/eval_hierarchy_report.json

if ($LASTEXITCODE -ne 0) {
    Write-Error "Hierarchy evaluation failed"
    exit 1
}

Write-Host "Hierarchy evaluation completed"
