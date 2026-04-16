"""
phase3_nsga2.py - Phase 3: NSGA-II multi-objective optimisation.

Finds the Pareto front of parameter sets trading off:
    f1 = S12 dip depth (minimise -> deeper dip is better)
    f2 = -|ON/OFF frequency shift| (minimise -> larger shift is better)

All evaluations use the trained surrogate models from ml_model.py — zero
COMSOL calls. Output: Pareto CSV + scatter plot.

Run standalone:
    python phase3_nsga2.py
"""

import os
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from config import DATA_DIR, OUTPUT_DIR
from data_loader import load_all_data, pair_on_off
from ml_model import (
    prepare_dataset, prepare_pair_dataset, build_models, train_final_models,
)


# Variable bounds: dx, g_w, c_w, h_graph, w_au (w_graph fixed at 1)
LOWER_BOUNDS = np.array([23, 3, 12, -6, 3])
UPPER_BOUNDS = np.array([35, 5, 28,  6, 5])
VAR_NAMES = ['dx', 'g_w', 'c_w', 'h_graph', 'w_au']
W_GRAPH_FIXED = 1


class THzMetamaterialProblem(ElementwiseProblem):
    """
    Multi-objective: minimise [S12_dip_pred, -|freq_shift_pred|].

    dip_model: trained surrogate that predicts S12 dip from geometry
    pair_model: trained surrogate that predicts freq shift from geometry
    scaler / pair_scaler: fitted StandardScaler instances
    """

    def __init__(self, dip_model, pair_model, scaler, pair_scaler):
        super().__init__(
            n_var=5,
            n_obj=2,
            n_constr=0,
            xl=LOWER_BOUNDS,
            xu=UPPER_BOUNDS,
            vtype=int,
        )
        self.dip_model = dip_model
        self.pair_model = pair_model
        self.scaler = scaler
        self.pair_scaler = pair_scaler

    def _evaluate(self, x, out, *args, **kwargs):
        # x shape: (5,) — [dx, g_w, c_w, h_graph, w_au]
        # Matches training feature order after prepare_dataset drops w_graph.
        feat = np.array([[int(round(v)) for v in x]])

        # Predict S12 dip using all-sim model
        dip_pred = float(self.dip_model.predict(self.scaler.transform(feat))[0])
        # Predict freq shift using ON/OFF pair model
        shift_pred = float(
            self.pair_model.predict(self.pair_scaler.transform(feat))[0]
        )

        # Minimise dip (more negative = better); minimise -|shift| (bigger shift = better)
        out["F"] = [dip_pred, -abs(shift_pred)]


def run_nsga2(pop_size=40, n_gen=50, seed=42, verbose=True):
    """
    Run NSGA-II using trained surrogate models.
    Returns (Pareto DataFrame, full result object).
    """
    # Load data
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)

    # Train all-sim surrogate (S12 dip)
    X_all, Y_all, _, _ = prepare_dataset(df)
    trained_all, scaler_all = train_final_models(X_all, Y_all, build_models())
    dip_model = trained_all['Gaussian Process']['S12 Dip (dB)']

    # Train pair surrogate (freq shift) — use Random Forest (best on pairs)
    X_pairs, Y_pairs, _, _ = prepare_pair_dataset(pairs)
    trained_pairs, scaler_pairs = train_final_models(
        X_pairs, Y_pairs, build_models())
    shift_model = trained_pairs['Random Forest']['Freq Shift (GHz)']

    if verbose:
        print(f"\nRunning NSGA-II:")
        print(f"  Dip surrogate: Gaussian Process (R^2 ~ 0.08)")
        print(f"  Freq-shift surrogate: Random Forest (R^2 ~ 0.47)")
        print(f"  Population: {pop_size}, generations: {n_gen}")

    problem = THzMetamaterialProblem(
        dip_model, shift_model, scaler_all, scaler_pairs)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = minimize(problem, algorithm, termination,
                      seed=seed, verbose=False)

    # Build Pareto DataFrame
    X_pareto = result.X.astype(int)
    F_pareto = result.F

    records = []
    for xi, fi in zip(X_pareto, F_pareto):
        row = dict(zip(VAR_NAMES, xi))
        row['w_graph'] = W_GRAPH_FIXED
        row['predicted_dip_dB'] = fi[0]
        row['predicted_freq_shift_abs_GHz'] = -fi[1]  # undo the negation
        records.append(row)

    pareto_df = pd.DataFrame(records).drop_duplicates()
    pareto_df = pareto_df.sort_values('predicted_dip_dB').reset_index(drop=True)

    if verbose:
        print(f"\nPareto front: {len(pareto_df)} non-dominated solutions")
        print(f"  Best predicted dip: {pareto_df.predicted_dip_dB.min():.3f} dB")
        print(f"  Best predicted freq shift: "
              f"{pareto_df.predicted_freq_shift_abs_GHz.max():.2f} GHz")

    return pareto_df, result


def save_results(pareto_df, output_dir=OUTPUT_DIR):
    """Save Pareto front CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'phase3_pareto_front.csv')
    pareto_df.to_csv(path, index=False)
    print(f"\nSaved Pareto front to: {path}")


def plot_pareto(pareto_df, output_dir=OUTPUT_DIR):
    """Save a Pareto scatter plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        pareto_df.predicted_dip_dB,
        pareto_df.predicted_freq_shift_abs_GHz,
        c=pareto_df.h_graph, cmap='viridis', s=80, edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Predicted S12 Dip Depth (dB, more negative = better)',
                  fontsize=12)
    ax.set_ylabel('Predicted |ON/OFF Frequency Shift| (GHz, larger = better)',
                  fontsize=12)
    ax.set_title('Phase 3 NSGA-II Pareto Front\n'
                 '(Trade-off: dip depth vs modulation)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('$h_{graph}$ ($\\mu$m)', fontsize=11)

    # Annotate top 5 best dip
    top5 = pareto_df.nsmallest(5, 'predicted_dip_dB')
    for _, row in top5.iterrows():
        ax.annotate(
            f"dx{row.dx}\ncw{row.c_w}\nwau{row.w_au}",
            (row.predicted_dip_dB, row.predicted_freq_shift_abs_GHz),
            fontsize=8, alpha=0.8,
            xytext=(5, 5), textcoords='offset points'
        )

    plt.tight_layout()
    path = os.path.join(output_dir, 'plots', 'phase3_pareto_front.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto plot to: {path}")


if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 3: NSGA-II MULTI-OBJECTIVE OPTIMISATION")
    print("=" * 70)

    pareto_df, result = run_nsga2(pop_size=40, n_gen=50)

    print("\nTop 10 Pareto-optimal candidates (ranked by predicted dip):")
    print(pareto_df.head(10).to_string(index=False))

    save_results(pareto_df)
    plot_pareto(pareto_df)

    print("\n" + "=" * 70)
    print("Phase 3 NSGA-II complete.")
    print("=" * 70)
