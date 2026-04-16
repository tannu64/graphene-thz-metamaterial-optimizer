"""
phase3_bayesopt.py - Phase 3: Bayesian Optimisation for THz metamaterial design.

Uses scikit-optimize's gp_minimize with Expected Improvement acquisition to
find parameter sets that maximise S12 dip depth (or a weighted multi-target
objective), using the trained Gaussian Process surrogate from ml_model.py.

Run standalone:
    python phase3_bayesopt.py
"""

import os
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real

from config import DATA_DIR, OUTPUT_DIR
from data_loader import load_all_data
from ml_model import prepare_dataset, build_models, train_final_models


# Search bounds derived from observed data ranges.
# w_graph is NOT in the search space because prepare_dataset() drops it
# (zero variance) before model training. The surrogate model therefore
# does not expect a w_graph column.
SEARCH_SPACE = [
    Integer(23, 35, name='dx'),
    Integer(3, 5, name='g_w'),
    Integer(12, 28, name='c_w'),
    Integer(-6, 6, name='h_graph'),
    Integer(3, 5, name='w_au'),
]

SEARCH_NAMES = ['dx', 'g_w', 'c_w', 'h_graph', 'w_au']
W_GRAPH_FIXED = 1  # used only for logging / output CSVs, not for prediction


def make_objective(trained_models, scaler, target_name='S12 Dip (dB)',
                   model_name='Gaussian Process'):
    """Create an objective function that minimises the chosen surrogate target."""
    model = trained_models[model_name][target_name]

    def objective(params):
        # params is [dx, g_w, c_w, h_graph, w_au] — matches the trained
        # feature order after w_graph was dropped by prepare_dataset().
        x = np.array(params).reshape(1, -1)
        x_scaled = scaler.transform(x)
        # For S12 dip, we want the MOST negative (deepest dip), so minimise prediction
        pred = model.predict(x_scaled)[0]
        return float(pred)

    return objective


def run_bayesian_optimisation(n_calls=60, random_state=42, target='S12 Dip (dB)',
                              model_name='Gaussian Process', verbose=True):
    """
    Run Bayesian Optimisation to find parameters minimising the chosen target.

    For S12 dip, minimising = deepest dip (best).
    Returns the skopt OptimizeResult and a DataFrame of top candidates.
    """
    # Load data and train surrogate
    df = load_all_data(DATA_DIR)
    X, Y, feat_cols, _ = prepare_dataset(df)
    trained, scaler = train_final_models(X, Y, build_models())

    # Build objective
    objective = make_objective(trained, scaler, target_name=target,
                               model_name=model_name)

    if verbose:
        print(f"\nRunning Bayesian Optimisation:")
        print(f"  Surrogate: {model_name}")
        print(f"  Target: {target} (minimise)")
        print(f"  Calls: {n_calls}")
        print(f"  Search space: {SEARCH_NAMES} (w_graph fixed at 1)")

    # Use gp_minimize with Expected Improvement
    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        acq_func='EI',           # Expected Improvement
        n_calls=n_calls,
        n_initial_points=10,
        random_state=random_state,
        verbose=False,
    )

    if verbose:
        print(f"\nBest predicted {target}: {result.fun:.4f}")
        best_params = dict(zip(SEARCH_NAMES, result.x))
        best_params['w_graph'] = W_GRAPH_FIXED
        print(f"  at: {best_params}")

    # Build DataFrame of all evaluated candidates (include w_graph for clarity)
    records = []
    for i, (params, val) in enumerate(zip(result.x_iters, result.func_vals)):
        row = dict(zip(SEARCH_NAMES, params))
        row['w_graph'] = W_GRAPH_FIXED
        row['predicted_' + target.split()[0].lower()] = val
        row['iteration'] = i + 1
        records.append(row)
    all_df = pd.DataFrame(records)

    # Top 10 by predicted target (ascending for dip, descending for shift)
    top_df = all_df.sort_values(
        'predicted_' + target.split()[0].lower()
    ).head(10).reset_index(drop=True)

    return result, top_df, all_df


def save_results(top_df, all_df, output_dir=OUTPUT_DIR):
    """Persist BO results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    top_path = os.path.join(output_dir, 'phase3_bo_top10.csv')
    all_path = os.path.join(output_dir, 'phase3_bo_all_iterations.csv')
    top_df.to_csv(top_path, index=False)
    all_df.to_csv(all_path, index=False)
    print(f"\nSaved top-10 to: {top_path}")
    print(f"Saved all iterations to: {all_path}")


def plot_convergence(result, output_dir=OUTPUT_DIR):
    """Save a convergence plot (running best vs. iteration)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vals = np.array(result.func_vals)
    running_best = np.minimum.accumulate(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(vals) + 1), vals, 'o-', alpha=0.3,
            label='Candidate prediction', color='#1565C0')
    ax.plot(range(1, len(running_best) + 1), running_best, 's-',
            linewidth=2, label='Running best', color='#C62828')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Predicted S12 Dip (dB)', fontsize=12)
    ax.set_title('Bayesian Optimisation Convergence\n(Expected Improvement, GP surrogate)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'plots', 'phase3_bo_convergence.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to: {path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE 3: BAYESIAN OPTIMISATION")
    print("=" * 70)

    result, top_df, all_df = run_bayesian_optimisation(
        n_calls=60, target='S12 Dip (dB)', model_name='Gaussian Process')

    print("\nTop 10 candidates (ranked by predicted dip depth):")
    print(top_df.to_string(index=False))

    save_results(top_df, all_df)
    plot_convergence(result)

    print("\n" + "=" * 70)
    print("Phase 3 Bayesian Optimisation complete.")
    print("=" * 70)
