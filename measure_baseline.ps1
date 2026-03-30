$RUNS = 3
$RUN_ID = "baseline"
$ROOT = "C:\Users\celin\Celina\repo\celestial-bodies-detection-thesis"
$TRAIN_DIR = "$ROOT\hub\examples\image_retraining"
$LOGFILE = "$ROOT\measurements\measurement_log.txt"
$csv = "$ROOT\measurements\f1_results.csv"
New-Item -ItemType Directory -Force -Path "$ROOT\measurements" | Out-Null

cd $TRAIN_DIR

# Clear previous logs and CSV
"" | Set-Content $LOGFILE
if (Test-Path $csv) { Remove-Item $csv }

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
Write-Host "`nDone. Results in $csv and $LOGFILE"

cd $ROOT