#!/usr/bin/env bash
set -e
set -x

RUNS=3
LOGFILE="measurement_log.txt"

cd hub/examples/image_retraining

# Clear previous log
echo -n "" > "$LOGFILE"

for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS"
    # Remove previous outputs for a clean run
    rm -f retrained_graph.pb retrained_labels.txt run_output.txt
    rm -rf bottlenecks

    python retrain.py \
        --bottleneck_dir=bottlenecks \
        --how_many_training_steps=500 \
        --model_dir=inception \
        --summaries_dir=training_summaries/basic \
        --output_graph=retrained_graph.pb \
        --output_labels=retrained_labels.txt \
        --image_dir=./training_data > run_output.txt 2>&1
    TIME=$(grep "Training Time:" run_output.txt | awk '{print $3}')
    echo "$TIME" >> "$LOGFILE"
done

AVG=$(awk '{sum+=$1} END {if (NR>0) print sum/NR}' "$LOGFILE")
echo "Average: $AVG" >> "$LOGFILE"
echo "All runs and average written to $LOGFILE"