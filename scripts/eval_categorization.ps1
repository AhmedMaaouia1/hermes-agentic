Write-Host "=== Evaluating categorization ==="

docker compose run --rm hermes `
  python src/evaluation/evaluate.py `
  --ground-truth data/ground_truth.json `
  --predictions logs/categorizer/categorizations.json `
  --output logs/categorizer/eval_categorization_report.json

if ($LASTEXITCODE -ne 0) {
    Write-Error "Categorization evaluation failed"
    exit 1
}

Write-Host "Categorization evaluation completed"
