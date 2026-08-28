param(
    [int]$Epochs = 15,
    [double]$TrainFraction = 1.0,
    [int]$Seed = 42,
    [int]$GradcamSamples = 200
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$resultPath = Join-Path $projectRoot "results\seed_$Seed"

python (Join-Path $projectRoot "src\experiment.py") `
    --data-dir (Join-Path $projectRoot "data") `
    --output-dir $resultPath `
    --epochs $Epochs `
    --train-fraction $TrainFraction `
    --seed $Seed `
    --gradcam-samples $GradcamSamples

