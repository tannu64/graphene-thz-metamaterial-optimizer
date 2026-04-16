"""
ml_model.py - Phase 2: Machine Learning models for S-parameter prediction.
Trains and evaluates multiple regression models on COMSOL simulation data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import DATA_DIR, OUTPUT_DIR
from data_loader import load_all_data, pair_on_off


def prepare_dataset(df, drop_zero_variance=True):
    """
    Build ML-ready X (features) and Y (targets) from simulation data.

    If drop_zero_variance=True (default), features with no variance in the
    dataset are removed. This avoids the misleading "zero importance"
    artefact and keeps the feature set honest about what the model actually
    has information on.

    Returns X, Y DataFrames and feature/target names.
    """
    feature_cols = ['dx', 'g_w', 'c_w', 'h_graph', 'w_graph', 'w_au']
    target_cols = ['s12_min', 's12_min_freq']

    X = df[feature_cols].copy()
    Y = df[target_cols].copy()
    Y.columns = ['S12 Dip (dB)', 'Dip Freq (GHz)']

    if drop_zero_variance:
        nunique = X.nunique()
        kept = [c for c in feature_cols if nunique[c] > 1]
        dropped = [c for c in feature_cols if nunique[c] <= 1]
        if dropped:
            print(f"  Dropping zero-variance features: {dropped}")
        X = X[kept]
        feature_cols = kept

    return X, Y, feature_cols, list(Y.columns)


def prepare_pair_dataset(pairs, drop_zero_variance=True):
    """
    Build ML-ready dataset from ON/OFF pairs for frequency shift prediction.
    Returns X, Y DataFrames.
    """
    feature_cols = ['dx', 'g_w', 'c_w', 'h_graph', 'w_graph', 'w_au']
    target_cols = ['freq_shift_ghz', 'avg_dip']

    X = pairs[feature_cols].copy()
    Y = pairs[target_cols].copy()
    Y.columns = ['Freq Shift (GHz)', 'Avg Dip (dB)']

    if drop_zero_variance:
        nunique = X.nunique()
        kept = [c for c in feature_cols if nunique[c] > 1]
        X = X[kept]
        feature_cols = kept

    return X, Y, feature_cols, list(Y.columns)


def build_models():
    """Return dict of model name -> model instance."""
    return {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        ),
        'Gaussian Process': GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(),
            n_restarts_optimizer=10, random_state=42, normalize_y=True
        ),
    }


def evaluate_models(X, Y, models):
    """
    Evaluate each model using Leave-One-Out cross-validation.
    LOO is ideal for small datasets — uses N-1 samples for training each fold.
    Returns results dict with predictions and metrics per target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo = LeaveOneOut()

    results = {}

    for model_name, model in models.items():
        results[model_name] = {}

        for target_name in Y.columns:
            y = Y[target_name].values

            # LOO cross-validated predictions
            y_pred = cross_val_predict(model, X_scaled, y, cv=loo)

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            results[model_name][target_name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'y_true': y,
                'y_pred': y_pred,
            }

    return results, scaler


def train_final_models(X, Y, models):
    """
    Train models on ALL data (for prediction use after evaluation).
    Returns dict of trained models and the scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    trained = {}
    for model_name, model in models.items():
        trained[model_name] = {}
        for target_name in Y.columns:
            y = Y[target_name].values
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_scaled, y)
            trained[model_name][target_name] = m

    return trained, scaler


def feature_importance(X, Y, feature_cols):
    """
    Compute feature importance using Random Forest.
    Returns DataFrame sorted by importance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    importances = {}
    for target_name in Y.columns:
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_scaled, Y[target_name].values)
        importances[target_name] = rf.feature_importances_

    imp_df = pd.DataFrame(importances, index=feature_cols)
    return imp_df


def print_results(results, dataset_label=""):
    """Print evaluation metrics in a readable format."""
    if dataset_label:
        print(f"\n{'='*70}")
        print(f"ML MODEL EVALUATION — {dataset_label}")
        print(f"{'='*70}")

    for model_name, targets in results.items():
        print(f"\n  {model_name}:")
        for target_name, metrics in targets.items():
            print(f"    {target_name}:")
            print(f"      R²   = {metrics['R2']:.4f}")
            print(f"      RMSE = {metrics['RMSE']:.4f}")
            print(f"      MAE  = {metrics['MAE']:.4f}")


def save_ml_report(results_all, results_pairs, imp_all, imp_pairs, output_dir,
                   n_all=None, n_pairs=None):
    """Save ML evaluation results to a text report."""
    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 70)
    add("MACHINE LEARNING MODEL EVALUATION REPORT")
    add("=" * 70)

    add(f"\n1. ALL SIMULATIONS ({n_all} samples, LOO cross-validation)")
    add("-" * 70)
    for model_name, targets in results_all.items():
        add(f"\n  {model_name}:")
        for target_name, m in targets.items():
            add(f"    {target_name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    if results_pairs:
        add(f"\n\n2. ON/OFF PAIRS ({n_pairs} pairs, LOO cross-validation)")
        add("-" * 70)
        for model_name, targets in results_pairs.items():
            add(f"\n  {model_name}:")
            for target_name, m in targets.items():
                add(f"    {target_name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    add(f"\n\n3. FEATURE IMPORTANCE (Random Forest)")
    add("-" * 70)
    add("\n  All simulations:")
    add(imp_all.to_string())
    if imp_pairs is not None:
        add("\n  ON/OFF pairs:")
        add(imp_pairs.to_string())

    # Best model summary
    add(f"\n\n4. BEST MODEL SUMMARY")
    add("-" * 70)
    for target_name in list(results_all.values())[0].keys():
        best_model = max(results_all.keys(), key=lambda m: results_all[m][target_name]['R2'])
        best_r2 = results_all[best_model][target_name]['R2']
        add(f"  {target_name}: {best_model} (R²={best_r2:.4f})")

    add("\n" + "=" * 70)

    report_path = os.path.join(output_dir, 'ml_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nML report saved to: {report_path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)

    print(f"Loaded {len(df)} simulations, {len(pairs)} ON/OFF pairs")

    # --- Dataset 1: All simulations (predict S12 dip + freq) ---
    X_all, Y_all, feat_cols, _ = prepare_dataset(df)
    models = build_models()

    print("\nEvaluating models on all simulations (LOO CV)...")
    results_all, _ = evaluate_models(X_all, Y_all, models)
    print_results(results_all, f"All Simulations ({len(df)} samples)")

    imp_all = feature_importance(X_all, Y_all, feat_cols)
    print(f"\nFeature Importance (all simulations):")
    print(imp_all.to_string())

    # --- Dataset 2: ON/OFF pairs (predict freq shift + avg dip) ---
    results_pairs = None
    imp_pairs = None
    if len(pairs) >= 5:
        X_pairs, Y_pairs, feat_cols_p, _ = prepare_pair_dataset(pairs)
        models_p = build_models()

        print("\nEvaluating models on ON/OFF pairs (LOO CV)...")
        results_pairs, _ = evaluate_models(X_pairs, Y_pairs, models_p)
        print_results(results_pairs, f"ON/OFF Pairs ({len(pairs)} samples)")

        imp_pairs = feature_importance(X_pairs, Y_pairs, feat_cols_p)
        print(f"\nFeature Importance (ON/OFF pairs):")
        print(imp_pairs.to_string())

    # --- Train final models on all data ---
    print("\nTraining final models on all data...")
    trained_all, scaler_all = train_final_models(X_all, Y_all, build_models())
    print("Done. Models ready for prediction.")

    # --- Save report ---
    save_ml_report(results_all, results_pairs, imp_all, imp_pairs, OUTPUT_DIR,
                   n_all=len(df), n_pairs=len(pairs))
