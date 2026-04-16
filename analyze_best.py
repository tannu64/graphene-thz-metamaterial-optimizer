"""
analyze_best.py - Dedicated best-performance analysis.

Answers the question: "at which input parameters does the device perform best,
and what does the neighbourhood of that best point look like?"

Outputs:
    output/best_performance_report.txt  — human-readable summary
    output/best_performance.csv         — top-5 rows across all metrics

Run: python analyze_best.py
"""

import os
import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR
from data_loader import load_all_data, pair_on_off


def summarise_best(df, pairs):
    """Return a dict of best-performing rows across multiple metrics."""
    best = {}

    # Best (deepest) S12 dip overall
    if len(df) > 0:
        idx = df['s12_min'].idxmin()
        best['deepest_s12_dip'] = df.loc[idx].to_dict()

    # Best ON-state dip (sigma = 0.3)
    on = df[df['sigma'] == 0.3]
    if len(on) > 0:
        idx = on['s12_min'].idxmin()
        best['deepest_on_state_dip'] = on.loc[idx].to_dict()

    # Best OFF-state dip (sigma = 1.2)
    off = df[df['sigma'] == 1.2]
    if len(off) > 0:
        idx = off['s12_min'].idxmin()
        best['deepest_off_state_dip'] = off.loc[idx].to_dict()

    # Largest ON/OFF frequency shift (modulation target)
    if len(pairs) > 0:
        idx = pairs['freq_shift_ghz'].idxmax()
        best['largest_freq_shift'] = pairs.loc[idx].to_dict()

        # Best average dip across ON/OFF pair
        idx = pairs['avg_dip'].idxmin()
        best['deepest_avg_pair_dip'] = pairs.loc[idx].to_dict()

    return best


def neighbourhood_analysis(df, pivot_row, radius=0.5):
    """
    Find simulations whose parameter vector is within 'radius' standard
    deviations of the pivot row's parameter vector, after standardisation.
    Returns a DataFrame with the pivot + its neighbours.
    """
    params = ['dx', 'g_w', 'c_w', 'h_graph', 'w_au']
    pivot = df[df.index == pivot_row.name][params].iloc[0]

    # Standardise
    mu = df[params].mean()
    sd = df[params].std().replace(0, 1)

    z_df = (df[params] - mu) / sd
    z_pivot = (pivot - mu) / sd

    # Euclidean distance in standardised space
    dist = np.sqrt(((z_df - z_pivot) ** 2).sum(axis=1))
    df = df.copy()
    df['z_distance'] = dist
    return df.sort_values('z_distance').head(5)


def sensitivity_near_best(df, best_row, params=None):
    """
    For each parameter, show how S12 dip changes when that parameter is
    varied while others are fixed at the best-row values.
    """
    if params is None:
        params = ['dx', 'g_w', 'c_w', 'h_graph', 'w_au']

    out = {}
    for p in params:
        # Fix all OTHER params at best-row values
        others = [x for x in params if x != p]
        mask = np.ones(len(df), dtype=bool)
        for o in others:
            mask &= (df[o] == best_row[o])
        # Also filter by sigma so we're comparing like with like
        if 'sigma' in best_row:
            mask &= (df['sigma'] == best_row['sigma'])
        sub = df[mask].copy()
        if len(sub) >= 2:
            sub = sub.sort_values(p)
            out[p] = sub[[p, 's12_min', 's12_min_freq']].to_dict('records')
    return out


def generate_report(df, pairs):
    lines = []
    add = lines.append

    add("=" * 72)
    add("BEST-PERFORMANCE ANALYSIS REPORT")
    add("=" * 72)
    add("")
    add(f"Dataset: {len(df)} simulations, {len(pairs)} ON/OFF pairs")
    add(f"Target dip depth: -20 dB")
    add("")

    # ------------------------------------------
    # 1. Best configurations
    # ------------------------------------------
    add("-" * 72)
    add("1. BEST CONFIGURATIONS (EACH METRIC)")
    add("-" * 72)

    best = summarise_best(df, pairs)
    for label, row in best.items():
        add(f"\n  {label.replace('_', ' ').upper()}:")
        for k, v in row.items():
            if isinstance(v, float):
                add(f"    {k:20s} = {v:.3f}")
            else:
                add(f"    {k:20s} = {v}")

    # ------------------------------------------
    # 2. Distance from target
    # ------------------------------------------
    add("")
    add("-" * 72)
    add("2. DISTANCE TO -20 dB TARGET")
    add("-" * 72)
    deepest = df['s12_min'].min()
    gap = abs(-20 - deepest)
    add(f"  Deepest observed dip: {deepest:.3f} dB")
    add(f"  Target:               -20.000 dB")
    add(f"  Gap:                  {gap:.3f} dB")
    add(f"  Percentage reached:   {(100 * deepest / -20):.1f}%")

    # ------------------------------------------
    # 3. Sensitivity around best point
    # ------------------------------------------
    add("")
    add("-" * 72)
    add("3. 1-D SENSITIVITY AROUND BEST ON-STATE DIP")
    add("-" * 72)

    best_row = df.loc[df[df['sigma'] == 0.3]['s12_min'].idxmin()]
    add(f"  Pivot row: dx={best_row.dx}, g_w={best_row.g_w}, "
        f"c_w={best_row.c_w}, h_graph={best_row.h_graph}, "
        f"w_au={best_row.w_au}, sigma={best_row.sigma}")
    add(f"  S12 dip at pivot: {best_row.s12_min:.3f} dB, "
        f"freq {best_row.s12_min_freq:.1f} GHz")

    sens = sensitivity_near_best(df, best_row)
    for p, rows in sens.items():
        if len(rows) >= 2:
            add(f"\n  Varying {p} (other params fixed):")
            add(f"    {p:>8s} | S12 dip (dB) | Dip freq (GHz)")
            for r in rows:
                add(f"    {r[p]:>8} | {r['s12_min']:>12.3f} | "
                    f"{r['s12_min_freq']:>14.1f}")

    # ------------------------------------------
    # 4. Nearest neighbours
    # ------------------------------------------
    add("")
    add("-" * 72)
    add("4. 5 NEAREST NEIGHBOURS OF THE BEST ON-STATE DIP")
    add("    (Euclidean distance in standardised parameter space)")
    add("-" * 72)

    nbrs = neighbourhood_analysis(df, best_row)
    add("\n  " + nbrs[['dx', 'g_w', 'c_w', 'h_graph', 'w_au', 'sigma',
                       's12_min', 's12_min_freq', 'z_distance']]
        .to_string().replace('\n', '\n  '))

    # ------------------------------------------
    # 5. Ranked top-5 table (all dimensions)
    # ------------------------------------------
    add("")
    add("-" * 72)
    add("5. TOP-5 RANKED BY DEEPEST S12 DIP (ANY SIGMA)")
    add("-" * 72)
    top5 = df.nsmallest(5, 's12_min')[
        ['dx', 'g_w', 'c_w', 'h_graph', 'w_au', 'sigma',
         's12_min', 's12_min_freq']
    ]
    add("\n  " + top5.to_string().replace('\n', '\n  '))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'best_performance.csv')
    top5.to_csv(csv_path, index=False)

    add("")
    add("=" * 72)

    report = '\n'.join(lines)
    path = os.path.join(OUTPUT_DIR, 'best_performance_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {path}")
    print(f"Top-5 CSV saved to: {csv_path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)
    generate_report(df, pairs)
