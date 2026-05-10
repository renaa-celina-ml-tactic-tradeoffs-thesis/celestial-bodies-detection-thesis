# This file is a script to run the image retraining process multiple times
# and measure the training time for each run, as well as calculate the averages
# scripts/measure_training_time.ps1

# to run this script, open PowerShell and execute: 
# .\measure_baseline.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

$RUNS = 2 # Number of times to run the training process
$RUN_ID = "baseline"
$TRAIN_DIR = Join-Path $ROOT "hub\examples\image_retraining" # Path to the retraining script, adjustable to your setup
$MEASUREMENTS_DIR = Join-Path $ROOT "measurements"
$LOGFILE = Join-Path $MEASUREMENTS_DIR "measurement_log.txt" # Log file to store training times and average
$csv = Join-Path $MEASUREMENTS_DIR "f1_results.csv" # CSV file to store F1, precision, recall, and average
$score_file = Join-Path $MEASUREMENTS_DIR "reliability_score.txt" # File to store the reliability score

New-Item -ItemType Directory -Force -Path $MEASUREMENTS_DIR | Out-Null

Set-Location $TRAIN_DIR

# Clear previous logs, CSV, and reliability score
"" | Set-Content $LOGFILE
if (Test-Path $csv) { Remove-Item $csv }
if (Test-Path $score_file) { Remove-Item $score_file }

$all_times = @()

for ($i = 1; $i -le $RUNS; $i++) {
    Write-Host "`n=== Run $i/$RUNS ==="

    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue bottlenecks, retrained_graph.pb, retrained_labels.txt

    python retrain.py `
        --image_dir=training_data `
        --bottleneck_dir=bottlenecks `
        --how_many_training_steps=500 `
        --output_graph=retrained_graph.pb `
        --output_labels=retrained_labels.txt `
        --test_dir=test_data `
        --run_id=$RUN_ID `
        --eval_runs=1 2>&1 | Tee-Object run_output.txt

    $TIME = Select-String "Training Time:" run_output.txt |
        ForEach-Object { ($_.Line -split "\s+")[2] }

    $all_times += [double]$TIME
    Add-Content $LOGFILE "Run ${i}: $TIME seconds"
    Write-Host "Training time for run ${i}: $TIME seconds"
}

$avg_time = [math]::Round(($all_times | Measure-Object -Average).Average, 4)
Add-Content $LOGFILE "Average training time: $avg_time seconds"
Write-Host "`nAverage training time: $avg_time seconds"

$rows = Import-Csv $csv
$avg_f1 = [math]::Round(($rows | ForEach-Object { [double]$_.f1_weighted } | Measure-Object -Average).Average, 4)
$avg_precision = [math]::Round(($rows | ForEach-Object { [double]$_.precision_weighted } | Measure-Object -Average).Average, 4)
$avg_recall = [math]::Round(($rows | ForEach-Object { [double]$_.recall_weighted } | Measure-Object -Average).Average, 4)

$avg_row = [PSCustomObject]@{
    timestamp          = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    run_id             = "${RUN_ID}_AVG"
    run_number         = 0
    f1_weighted        = $avg_f1
    precision_weighted = $avg_precision
    recall_weighted    = $avg_recall
}

$avg_row | Export-Csv $csv -Append -NoTypeInformation

Write-Host "Average F1: $avg_f1 | Precision: $avg_precision | Recall: $avg_recall"

# --- Read and log the reliability score produced by retrain.py ---
if (Test-Path $score_file) {
    $reliability_lines = Get-Content $score_file
    $consistency_line = $reliability_lines | Where-Object { $_ -match "^Consistency Score:" }
    $instability_line = $reliability_lines | Where-Object { $_ -match "^Mean Instability \(" }
    $num_images_line  = $reliability_lines | Where-Object { $_ -match "^Number of test images evaluated:" }

    if ($consistency_line) {
        $consistency_score = ($consistency_line -split ":\s*")[-1].Trim()
        $instability_score = ($instability_line -split ":\s*")[-1].Trim()
        $num_images        = ($num_images_line  -split ":\s*")[-1].Trim()

        Add-Content $LOGFILE "Consistency Score: $consistency_score"
        Add-Content $LOGFILE "Mean Instability: $instability_score"
        Add-Content $LOGFILE "Test images evaluated: $num_images"

        Write-Host "`nReliability (Output Consistency Score): $consistency_score"
        Write-Host "Mean Instability: $instability_score"
        Write-Host "Test images evaluated: $num_images"
    } else {
        Write-Host "WARNING: reliability_score.txt found but could not parse Consistency Score."
        Add-Content $LOGFILE "WARNING: Could not parse reliability_score.txt"
    }
} else {
    Write-Host "WARNING: reliability_score.txt not found. Reliability score may not have been generated."
    Add-Content $LOGFILE "WARNING: reliability_score.txt not found."
}
# --- End reliability score logging ---

Write-Host "`nDone. Results in $csv, $LOGFILE, and $score_file"

Set-Location $ROOT