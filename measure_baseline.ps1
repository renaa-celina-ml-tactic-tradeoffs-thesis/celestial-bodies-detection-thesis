# This file is a script to run the image retraining process multiple times
# and measure the training time for each run, as well as calculate the averages
# scripts/measure_training_time.ps1

# to run this script, open PowerShell and execute: 
# .\measure_baseline.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

$RUNS = 10 # Number of times to run the training process
$RUN_ID = "baseline"
$TRAIN_DIR = Join-Path $ROOT "hub\examples\image_retraining" # Path to the retraining script, adjustable to your setup
$MEASUREMENTS_DIR = Join-Path $ROOT "measurements"
$LOGFILE = Join-Path $MEASUREMENTS_DIR "measurement_log.txt" # Log file to store training times and average
$csv = Join-Path $MEASUREMENTS_DIR "f1_results.csv" # CSV file to store F1, precision, recall, and average
$score_file = Join-Path $MEASUREMENTS_DIR "reliability_score.txt" # File to store the reliability score
$cc_file = Join-Path $MEASUREMENTS_DIR "cc_score.txt" # File to store cyclomatic complexity results

New-Item -ItemType Directory -Force -Path $MEASUREMENTS_DIR | Out-Null

Set-Location $TRAIN_DIR

# Clear previous logs and CSV
"" | Set-Content $LOGFILE
if (Test-Path $csv) { Remove-Item $csv }
if (Test-Path $score_file) { Remove-Item $score_file }

# === Cyclomatic Complexity Measurement ===
Write-Host "`n=== Measuring Cyclomatic Complexity ==="
$cc_raw = python -m radon cc retrain.py -s 2>&1

# Save full individual function breakdown to file
$cc_raw | Out-File $cc_file -Encoding utf8

# Compute weighted average CC inline via Python one-liner
$weighted_avg = python -c @"
import re, sys

output = '''$($cc_raw -join "`n")'''

pattern = re.compile(r'F\s+(\d+):\d+\s+\S+\s+-\s+[A-F]\s+\((\d+)\)')
functions = [(int(m.group(1)), int(m.group(2))) for m in pattern.finditer(output)]
functions.sort()

if not functions:
    print('N/A')
    sys.exit()

spans = []
for i, (line, cc) in enumerate(functions):
    spans.append(functions[i+1][0] - line if i+1 < len(functions) else 20)

weighted = sum(cc * s for (_, cc), s in zip(functions, spans)) / sum(spans)
print(f'{weighted:.4f}')
"@ 2>&1

if ($weighted_avg -and $weighted_avg -ne 'N/A') {
    Write-Host "CC Weighted Average: $weighted_avg"
    Add-Content $LOGFILE "CC Weighted Average: $weighted_avg"
    Add-Content $cc_file "`nWeighted Average CC: $weighted_avg"
} else {
    Write-Host "CC measurement failed or produced no output."
    Add-Content $LOGFILE "Cyclomatic Complexity: measurement failed"
}
# === End CC Measurement ===

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
        --flip_left_right `
        --random_crop=10 `
        --random_scale=10 `
        --random_brightness=10 `
        --augmentations_per_image=3 `
        --eval_runs=1 2>&1 | Tee-Object run_output.txt
        

    $TIME = Select-String "Training Time:" run_output.txt |
        Select-Object -First 1 |
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

# --- Reliability Score: F1 consistency across runs ---
$f1_values = $rows | ForEach-Object { [double]$_.f1_weighted }
$mean_f1 = ($f1_values | Measure-Object -Average).Average
$variance = ($f1_values | ForEach-Object { [math]::Pow($_ - $mean_f1, 2) } | Measure-Object -Average).Average
$std_dev = [math]::Round([math]::Sqrt($variance), 4)
$consistency_score = [math]::Round(1.0 - $std_dev, 4)

"Reliability Metric: Output Consistency Score" | Set-Content $score_file
"Consistency Score: $consistency_score" | Add-Content $score_file
"Mean Instability (F1 std dev across runs): $std_dev" | Add-Content $score_file
"Number of runs: $($rows.Count)" | Add-Content $score_file

Add-Content $LOGFILE "Consistency Score: $consistency_score"
Add-Content $LOGFILE "F1 Std Dev (instability): $std_dev"

Write-Host "`nReliability (Output Consistency Score): $consistency_score"
Write-Host "F1 Std Dev across runs: $std_dev"
# --- End reliability score ---

Write-Host "`nDone. Results in $csv, $LOGFILE, and $score_file"

Set-Location $ROOT