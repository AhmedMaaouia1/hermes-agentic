Write-Host "=== Evaluating reviewer output ==="

docker compose run --rm hermes `
  python src/evaluation/evaluate.py `
  --ground-truth data/ground_truth.json `
  --predictions logs/reviewer/review.json `
  --output logs/reviewer/eval_reviewer_report.json

if ($LASTEXITCODE -ne 0) {
    Write-Error "Reviewer evaluation failed"
    exit 1
}

Write-Host "Reviewer evaluation completed"