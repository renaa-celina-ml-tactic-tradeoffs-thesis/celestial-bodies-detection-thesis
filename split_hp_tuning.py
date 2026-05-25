#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

# ----- INPUTS -----
# Point these to wherever your combined HP files currently live
SRC_F1  = Path("data/hyperparameter_tuning/f1_results.csv")
SRC_LOG = Path("data/hyperparameter_tuning/measurement_log.txt")

# Root where your tactics live (same root you pass to qa_statistical_tests.py)
OUT_ROOT = Path("data")

# ----- CONFIG DEFINITIONS -----
CONFIGS = [
    ("hp_tuning_config_a", "lr0.01_steps4000", 0),
    ("hp_tuning_config_b", "lr0.001_steps4000", 1),
    ("hp_tuning_config_c", "lr0.05_steps4000", 2),
    ("hp_tuning_config_d", "lr0.01_steps2000", 3),
    ("hp_tuning_config_e", "lr0.01_steps6000", 4),
]

def split_times(src_log: Path):
    """Return list of 5 lists, each with 10 floats."""
    cols = [[] for _ in range(5)]
    with src_log.open("r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*Run\s+(\d+):\s+(.*?)\s+seconds", line.strip())
            if not m:
                continue
            nums = [float(x) for x in m.group(2).split()]
            if len(nums) != 5:
                raise ValueError(f"Expected 5 numbers per run, got {len(nums)} in line: {line}")
            for i, x in enumerate(nums):
                cols[i].append(x)

    for i, c in enumerate(cols):
        if len(c) != 10:
            raise ValueError(f"Expected 10 runs for column {i+1}, got {len(c)}")
    return cols

def write_measurement_log(out_path: Path, times):
    """Write in the format your parser expects: 'Run i: X seconds'."""
    with out_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(times, start=1):
            f.write(f"Run {i}: {t:.4f} seconds\n")

def main():
    # Load and filter F1
    df = pd.read_csv(SRC_F1, encoding="utf-8-sig")
    df["run_id"] = df["run_id"].astype(str)

    # Drop AVG row if present
    df = df[~df["run_id"].str.endswith("_AVG", na=False)]

    # Split times
    time_cols = split_times(SRC_LOG)

    # Write per-config folders
    for folder_name, run_id, col_idx in CONFIGS:
        out_dir = OUT_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # F1: keep only this config's 10 rows
        df_cfg = df[df["run_id"] == run_id].copy()
        if len(df_cfg) != 10:
            raise ValueError(f"{folder_name}: expected 10 F1 rows for {run_id}, got {len(df_cfg)}")

        df_cfg.to_csv(out_dir / "f1_results.csv", index=False, encoding="utf-8")

        # Time: write single-column format
        write_measurement_log(out_dir / "measurement_log.txt", time_cols[col_idx])

        print(f"Wrote {folder_name}/f1_results.csv and measurement_log.txt")

if __name__ == "__main__":
    main()