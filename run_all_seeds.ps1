$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$seeds = 42, 123, 2026

foreach ($seed in $seeds) {
    & (Join-Path $projectRoot "run_experiment.ps1") -Seed $seed
    if ($LASTEXITCODE -ne 0) {
        throw "A execução com a semente $seed falhou."
    }
}

python (Join-Path $projectRoot "src\aggregate_results.py") `
    --results-dir (Join-Path $projectRoot "results") `
    --output-dir (Join-Path $projectRoot "results\aggregate")

