#!/usr/bin/env python3
"""
qa_statistical_tests.py
────────────────────────────────────────────────────────────────────────────
Mann-Whitney U significance testing for architectural tactic QA metrics.

Metrics tested:
  • F1-score        (System Accuracy)      — 10 observations per tactic
  • Training time   (Resource Efficiency)  — 10 observations per tactic
  • Cyclomatic CC   (Maintainability)      — per-function values from cc_score.txt
  • OC Score        (Reliability)          — single aggregate; descriptive only

For F1 and Time, Bonferroni correction is applied across all tactics
tested against the baseline (n_tests = number of tactics per metric).
For CC, the per-function distributions are compared, which gives a
distributional test rather than a point estimate. OC is reported
descriptively since it is a single deterministic value per tactic.

Cliff's delta is reported as the effect size alongside each U-test.
  |δ| < 0.147  →  negligible
  |δ| < 0.330  →  small
  |δ| < 0.474  →  medium
  |δ| ≥ 0.474  →  large

Usage
─────
  Organise each tactic's output files in its own subfolder under a root
  data directory, with 'baseline' as the reference folder name:

    data/
      baseline/
          f1_results.csv
          measurement_log.txt
          reliability_score.txt
          cc_score.txt
      auto_data_reduction/
          f1_results.csv
          ...
      data_preprocessing/
          ...

  Then run:
      python qa_statistical_tests.py data/

  Optional flags:
      --alpha 0.05       significance level (default 0.05)
      --no-csv           skip writing statistical_results.csv

Requirements: numpy, scipy  (pip install numpy scipy)
"""

import os
import re
import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_f1_csv(path: Path) -> np.ndarray:
    """
    Parse F1 scores from f1_results.csv.
    Expected columns (any order): timestamp, run_id, run_number,
    f1_weighted, precision_weighted, recall_weighted.
    Rows whose run_id ends with '_AVG' are skipped.
    """
    f1_scores = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip the average summary row
            run_id = row.get('run_id', '').strip().strip('"')
            if run_id.endswith('_AVG'):
                continue
            # Try common column name variants
            for key in ['f1_weighted', 'F1-Score', 'F1_Score', 'F1', 'f1']:
                if key in row:
                    val = row[key].strip().strip('"')
                    try:
                        f1_scores.append(float(val))
                    except ValueError:
                        pass
                    break
    return np.array(f1_scores)


def parse_training_times(path: Path) -> np.ndarray:
    """
    Parse per-run training times (seconds) from measurement_log.txt.
    Looks for lines matching:  Run N: XX.XXXX seconds
    """
    times = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\s*Run\s+\d+:\s+([\d.]+)\s+seconds', line)
            if m:
                times.append(float(m.group(1)))
    return np.array(times)


def parse_oc_score(path: Path) -> float | None:
    """
    Parse the single OC score from reliability_score.txt.
    Looks for a line containing:  Consistency Score: X.XXXX
    """
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Consistency Score:\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
    return None


def parse_cc_function_values(path: Path) -> np.ndarray:
    """
    Parse per-function integer CC values from cc_score.txt.
    Looks for lines ending in:   - Grade (CC_value)
    e.g.  F 792:0 main - C (12)
    """
    cc_values = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'\((\d+)\)\s*$', line.strip())
            if m:
                cc_values.append(int(m.group(1)))
    return np.array(cc_values)


def parse_weighted_cc(path: Path) -> float | None:
    """Parse the weighted average CC summary line from cc_score.txt."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Weighted Average CC:\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size for two independent groups.
    Positive → x tends to be larger than y.
    Negative → x tends to be smaller than y.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float('nan')
    dominance = sum(
        (1 if xi > yj else -1 if xi < yj else 0)
        for xi in x for yj in y
    )
    return dominance / (n1 * n2)


def interpret_cliff(d: float) -> str:
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.330: return "small"
    if ad < 0.474: return "medium"
    return "large"


def run_mwu(baseline: np.ndarray, tactic: np.ndarray) -> dict:
    """
    Two-sided Mann-Whitney U test.
    Returns raw p-value; Bonferroni correction applied later in batch.
    """
    if len(baseline) < 2 or len(tactic) < 2:
        return None
    u_stat, p_val = stats.mannwhitneyu(baseline, tactic, alternative='two-sided')
    d = cliff_delta(tactic, baseline)
    return {
        'U':           u_stat,
        'p_raw':       p_val,
        'p_adj':       None,       # filled in after Bonferroni
        'sig':         None,       # filled in after Bonferroni
        'cliff_delta': d,
        'effect_size': interpret_cliff(d),
        'n_baseline':  len(baseline),
        'n_tactic':    len(tactic),
    }


def apply_bonferroni(results: dict[str, dict], alpha: float) -> dict[str, dict]:
    """
    Apply Bonferroni correction across all tactics for a single metric.
    Modifies results in-place, returns the dict.
    """
    valid = {k: v for k, v in results.items() if v is not None}
    n_tests = len(valid)
    for r in valid.values():
        r['p_adj'] = min(r['p_raw'] * n_tests, 1.0)
        r['sig'] = r['p_adj'] < alpha
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_tactic(folder: Path) -> dict:
    data = {
        'name':        folder.name,
        'f1':          None,
        'times':       None,
        'oc':          None,
        'cc_values':   None,
        'cc_weighted': None,
    }
    f1_path  = folder / 'f1_results.csv'
    log_path = folder / 'measurement_log.txt'
    rel_path = folder / 'reliability_score.txt'
    cc_path  = folder / 'cc_score.txt'

    if f1_path.exists():
        data['f1']     = parse_f1_csv(f1_path)
    if log_path.exists():
        data['times']  = parse_training_times(log_path)
    if rel_path.exists():
        data['oc']     = parse_oc_score(rel_path)
    if cc_path.exists():
        data['cc_values']   = parse_cc_function_values(cc_path)
        data['cc_weighted'] = parse_weighted_cc(cc_path)

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

WIDTH = 72

def sep(char='─'):
    print(char * WIDTH)

def pct_change(baseline_mean, tactic_mean):
    if baseline_mean == 0:
        return float('nan')
    return (tactic_mean - baseline_mean) / baseline_mean * 100


def print_metric_block(label: str, unit: str, higher_is_better: bool,
                       baseline_arr: np.ndarray, tactic_arr: np.ndarray,
                       result: dict):
    b_mean = np.mean(baseline_arr)
    t_mean = np.mean(tactic_arr)
    delta  = pct_change(b_mean, t_mean)
    better = (t_mean > b_mean) == higher_is_better
    arrow  = ("▲ better" if better else "▼ worse") + \
             ("  ← lower is better" if not higher_is_better else "")

    print(f"\n  [{label}]  unit: {unit}  |  "
          f"{'higher' if higher_is_better else 'lower'} is better")
    print(f"    baseline mean  = {b_mean:.6f}  (n={result['n_baseline']})")
    print(f"    tactic mean    = {t_mean:.6f}  (n={result['n_tactic']})")
    print(f"    Δ              = {delta:+.3f}%  {arrow}")
    print(f"    U statistic    = {result['U']:.1f}")
    print(f"    p (raw)        = {result['p_raw']:.6f}")
    print(f"    p (Bonferroni) = {result['p_adj']:.6f}  "
          f"{'✓ SIGNIFICANT' if result['sig'] else '✗ not significant'}")
    print(f"    Cliff's δ      = {result['cliff_delta']:+.4f}  "
          f"({result['effect_size']} effect)")


def print_oc_block(baseline_oc: float, tactic_oc: float):
    delta = pct_change(baseline_oc, tactic_oc)
    better = tactic_oc > baseline_oc
    print(f"\n  [OC Score — Reliability]  higher is better")
    print(f"    baseline OC    = {baseline_oc:.4f}")
    print(f"    tactic OC      = {tactic_oc:.4f}")
    print(f"    Δ              = {delta:+.3f}%  "
          f"{'▲ better' if better else '▼ worse'}")
    print(f"    ⚠  OC is a single aggregate value per tactic.")
    print(f"       No hypothesis test is applicable (n=1).")
    print(f"       Interpret as descriptive only.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(data_dir: str, alpha: float = 0.05, write_csv: bool = True):
    root = Path(data_dir)

    baseline_path = root / 'baseline'
    if not baseline_path.exists():
        sys.exit(f"ERROR: expected a 'baseline' subfolder in {root}")

    baseline = load_tactic(baseline_path)
    tactics  = sorted(
        [load_tactic(d) for d in root.iterdir()
         if d.is_dir() and d.name != 'baseline'],
        key=lambda t: t['name']
    )
    if not tactics:
        sys.exit("ERROR: no tactic subfolders found.")

    # ── Run all tests (raw) ──────────────────────────────────────────────────
    f1_results   = {}
    time_results = {}
    cc_results   = {}

    for t in tactics:
        name = t['name']
        if baseline['f1'] is not None and t['f1'] is not None and \
                len(t['f1']) > 0:
            f1_results[name] = run_mwu(baseline['f1'], t['f1'])
        if baseline['times'] is not None and t['times'] is not None and \
                len(t['times']) > 0:
            time_results[name] = run_mwu(baseline['times'], t['times'])
        if baseline['cc_values'] is not None and t['cc_values'] is not None and \
                len(t['cc_values']) > 1:
            cc_results[name] = run_mwu(baseline['cc_values'], t['cc_values'])

    # ── Apply Bonferroni per metric ──────────────────────────────────────────
    apply_bonferroni(f1_results,   alpha)
    apply_bonferroni(time_results, alpha)
    apply_bonferroni(cc_results,   alpha)

    # ── Per-tactic report ────────────────────────────────────────────────────
    sep('═')
    print("  QA STATISTICAL SIGNIFICANCE ANALYSIS")
    print("  Mann-Whitney U (two-sided) + Bonferroni correction + Cliff's δ")
    print(f"  α = {alpha}  |  n_tactics = {len(tactics)}")
    sep('═')

    for t in tactics:
        name = t['name']
        print()
        sep()
        print(f"  TACTIC: {name.replace('_', ' ').upper()}")
        sep()

        if name in f1_results and f1_results[name]:
            print_metric_block(
                "F1-Score — System Accuracy", "F1 (weighted)", True,
                baseline['f1'], t['f1'], f1_results[name])

        if name in time_results and time_results[name]:
            print_metric_block(
                "Training Time — Resource Efficiency", "seconds", False,
                baseline['times'], t['times'], time_results[name])

        if name in cc_results and cc_results[name]:
            print_metric_block(
                "Cyclomatic Complexity — Maintainability",
                "CC per function", False,
                baseline['cc_values'], t['cc_values'], cc_results[name])

        if baseline['oc'] is not None and t['oc'] is not None:
            print_oc_block(baseline['oc'], t['oc'])

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    sep('═')
    print("  SUMMARY TABLE  (Bonferroni-corrected, α = {})".format(alpha))
    sep('═')
    hdr = f"  {'Tactic':<30}  {'F1':^18}  {'Time':^18}  {'CC':^18}"
    print(hdr)
    sep()

    for t in tactics:
        name  = t['name']
        cells = []
        for rdict, b_arr, t_arr, hib in [
            (f1_results,   baseline['f1'],       t['f1'],       True),
            (time_results, baseline['times'],     t['times'],    False),
            (cc_results,   baseline['cc_values'], t['cc_values'], False),
        ]:
            r = rdict.get(name)
            if r and b_arr is not None and t_arr is not None and len(t_arr) > 0:
                b_mean = np.mean(b_arr)
                t_mean = np.mean(t_arr)
                delta  = pct_change(b_mean, t_mean)
                sig    = "✓*" if r['sig'] else "  "
                better = (t_mean > b_mean) == hib
                arrow  = "▲" if better else "▼"
                cells.append(f"{sig} {delta:+.2f}% {arrow} δ={r['cliff_delta']:+.3f}")
            else:
                cells.append("N/A")
        print(f"  {name.replace('_',' '):<30}  "
              f"{cells[0]:^18}  {cells[1]:^18}  {cells[2]:^18}")

    print()
    print("  ✓* = statistically significant after Bonferroni correction")
    print("  ▲ = improvement  ▼ = degradation  (direction relative to baseline)")
    print("  OC Score omitted from table — descriptive only (no per-run data)")

    # ── OC descriptive summary ───────────────────────────────────────────────
    print()
    sep()
    print("  OC SCORE DESCRIPTIVE SUMMARY  (no hypothesis test applicable)")
    sep()
    print(f"  {'Tactic':<30}  {'Baseline OC':>12}  {'Tactic OC':>10}  {'Δ':>8}")
    sep('·')
    for t in tactics:
        if baseline['oc'] is not None and t['oc'] is not None:
            delta = pct_change(baseline['oc'], t['oc'])
            arrow = "▲" if t['oc'] > baseline['oc'] else "▼"
            print(f"  {t['name'].replace('_',' '):<30}  "
                  f"{baseline['oc']:>12.4f}  {t['oc']:>10.4f}  "
                  f"{delta:>+7.3f}% {arrow}")
    print()

    # ── CSV export ───────────────────────────────────────────────────────────
    if write_csv:
        csv_path = root / 'statistical_results.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Tactic',
                # F1
                'F1_baseline_mean', 'F1_tactic_mean', 'F1_pct_change',
                'F1_U', 'F1_p_raw', 'F1_p_bonferroni',
                'F1_significant', 'F1_cliffs_delta', 'F1_effect_size',
                # Time
                'Time_baseline_mean', 'Time_tactic_mean', 'Time_pct_change',
                'Time_U', 'Time_p_raw', 'Time_p_bonferroni',
                'Time_significant', 'Time_cliffs_delta', 'Time_effect_size',
                # CC
                'CC_baseline_fn_mean', 'CC_tactic_fn_mean', 'CC_pct_change',
                'CC_U', 'CC_p_raw', 'CC_p_bonferroni',
                'CC_significant', 'CC_cliffs_delta', 'CC_effect_size',
                # OC
                'OC_baseline', 'OC_tactic', 'OC_pct_change', 'OC_note',
            ])
            for t in tactics:
                name = t['name']
                row  = [name.replace('_', ' ')]

                for rdict, b_arr, t_arr in [
                    (f1_results,   baseline['f1'],       t['f1']),
                    (time_results, baseline['times'],     t['times']),
                    (cc_results,   baseline['cc_values'], t['cc_values']),
                ]:
                    r = rdict.get(name)
                    if r and b_arr is not None and t_arr is not None and len(t_arr) > 0:
                        b_mean = np.mean(b_arr)
                        t_mean = np.mean(t_arr)
                        delta  = pct_change(b_mean, t_mean)
                        row += [f"{b_mean:.6f}", f"{t_mean:.6f}", f"{delta:+.4f}%",
                                f"{r['U']:.1f}", f"{r['p_raw']:.6f}",
                                f"{r['p_adj']:.6f}", str(r['sig']),
                                f"{r['cliff_delta']:+.4f}", r['effect_size']]
                    else:
                        row += ['N/A'] * 9

                if baseline['oc'] is not None and t['oc'] is not None:
                    delta = pct_change(baseline['oc'], t['oc'])
                    row += [f"{baseline['oc']:.4f}", f"{t['oc']:.4f}",
                            f"{delta:+.4f}%", "descriptive only"]
                else:
                    row += ['N/A', 'N/A', 'N/A', 'N/A']

                writer.writerow(row)

        print(f"  Full results saved to: {csv_path}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mann-Whitney U significance testing for architectural tactic QA metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'data_dir',
        help='Root folder containing baseline/ and one subfolder per tactic'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='Significance level (default: 0.05)'
    )
    parser.add_argument(
        '--no-csv', action='store_true',
        help='Skip writing statistical_results.csv'
    )
    args = parser.parse_args()
    run_analysis(args.data_dir, alpha=args.alpha, write_csv=not args.no_csv)