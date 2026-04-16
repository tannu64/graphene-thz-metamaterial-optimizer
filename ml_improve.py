"""
ml_improve.py - Attempt to improve ML model reliability beyond baseline.

Strategies tried:
1. Hyperparameter tuning via GridSearchCV for RF and GB
2. Alternative GP kernels (Matern 3/2, Matern 5/2) that are more flexible than RBF
3. Target transformation (standardised targets)
4. Internal validation via repeated K-fold CV and bootstrap confidence intervals

This module does NOT replace ml_model.py; it provides a separate diagnostic
and improvement pipeline that writes results to output/ml_improve_report.txt.

Run:
    python ml_improve.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneOut, RepeatedKFold, cross_val_predict, cross_val_score,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel, WhiteKernel,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample

from config import DATA_DIR, OUTPUT_DIR
from data_loader import load_all_data, pair_on_off
from ml_model import prepare_dataset, prepare_pair_dataset


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
# Hyperparameter grids (small — dataset is small)
# ---------------------------------------------------------------
RF_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5],
    'min_samples_leaf': [1, 2, 3],
}

GB_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_samples_leaf': [1, 2, 3],
}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def tune_rf(X, y):
    """Find best RF hyperparameters using LOO CV."""
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(
        rf, RF_GRID, cv=LeaveOneOut(), scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def tune_gb(X, y):
    """Find best GB hyperparameters using LOO CV."""
    gb = GradientBoostingRegressor(random_state=42)
    grid = GridSearchCV(
        gb, GB_GRID, cv=LeaveOneOut(), scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(model, X_scaled, y, cv):
    """Compute LOO-CV metrics for a model."""
    preds = cross_val_predict(model, X_scaled, y, cv=cv)
    return {
        'R2': r2_score(y, preds),
        'RMSE': np.sqrt(mean_squared_error(y, preds)),
        'MAE': mean_absolute_error(y, preds),
    }


def gp_with_kernel(kernel):
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10,
        random_state=42, normalize_y=True,
    )


def try_gp_kernels(X_scaled, y):
    """Compare GP with different kernels."""
    kernels = {
        'RBF': ConstantKernel() * RBF() + WhiteKernel(),
        'Matern3/2': ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        'Matern5/2': ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
    }
    loo = LeaveOneOut()
    out = {}
    for name, k in kernels.items():
        m = gp_with_kernel(k)
        out[name] = evaluate_model(m, X_scaled, y, loo)
    return out


def kfold_repeated(model, X_scaled, y, n_splits=5, n_repeats=10):
    """Repeated K-fold CV for a stability estimate."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    return {
        'R2_mean': float(np.mean(scores)),
        'R2_std': float(np.std(scores)),
        'R2_p5': float(np.percentile(scores, 5)),
        'R2_p95': float(np.percentile(scores, 95)),
        'n_scores': len(scores),
    }


def bootstrap_r2(model, X_scaled, y, n_bootstraps=200, seed=42):
    """Bootstrap resampling to estimate R² confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    loo = LeaveOneOut()
    base_preds = cross_val_predict(model, X_scaled, y, cv=loo)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        scores.append(r2_score(y[idx], base_preds[idx]))

    scores = np.array(scores)
    return {
        'R2_bootstrap_mean': float(np.mean(scores)),
        'R2_bootstrap_std': float(np.std(scores)),
        'R2_CI_low': float(np.percentile(scores, 2.5)),
        'R2_CI_high': float(np.percentile(scores, 97.5)),
    }


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def run_improvement_analysis():
    """Run the full improvement and validation analysis."""
    lines = []
    add = lines.append

    add("=" * 72)
    add("ML IMPROVEMENT & INTERNAL VALIDATION REPORT")
    add("=" * 72)
    add("")

    # --- Load data ---
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)

    add(f"Dataset: {len(df)} simulations, {len(pairs)} ON/OFF pairs\n")

    # --- All simulations ---
    X, Y, feat_cols, _ = prepare_dataset(df)
    add(f"Active features (after zero-variance removal): {feat_cols}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo = LeaveOneOut()

    for target in Y.columns:
        y = Y[target].values
        add("-" * 72)
        add(f"TARGET: {target}")
        add("-" * 72)

        # 1. Tuned Random Forest
        add("\n[1] Tuned Random Forest (GridSearchCV over RF_GRID):")
        best_rf, best_rf_params = tune_rf(X_scaled, y)
        m = evaluate_model(best_rf, X_scaled, y, loo)
        add(f"    Best params: {best_rf_params}")
        add(f"    R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

        # 2. Tuned Gradient Boosting
        add("\n[2] Tuned Gradient Boosting (GridSearchCV over GB_GRID):")
        best_gb, best_gb_params = tune_gb(X_scaled, y)
        m = evaluate_model(best_gb, X_scaled, y, loo)
        add(f"    Best params: {best_gb_params}")
        add(f"    R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

        # 3. GP with different kernels
        add("\n[3] GP kernel comparison:")
        gp_results = try_gp_kernels(X_scaled, y)
        for k, m in gp_results.items():
            add(f"    {k:12s}: R²={m['R2']:.4f}, "
                f"RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

        # 4. Internal validation on best RF
        add("\n[4] Internal validation on best RF (repeated 5-fold × 10):")
        kf = kfold_repeated(best_rf, X_scaled, y)
        add(f"    R² (mean ± std):  {kf['R2_mean']:.3f} ± {kf['R2_std']:.3f}")
        add(f"    R² (5th–95th):    [{kf['R2_p5']:.3f}, {kf['R2_p95']:.3f}]")

        add("\n[5] Bootstrap 95% CI on LOO R² (best RF, 200 resamples):")
        bs = bootstrap_r2(best_rf, X_scaled, y)
        add(f"    R² mean:     {bs['R2_bootstrap_mean']:.3f} "
            f"± {bs['R2_bootstrap_std']:.3f}")
        add(f"    95% CI:      [{bs['R2_CI_low']:.3f}, "
            f"{bs['R2_CI_high']:.3f}]")

        add("")

    # --- ON/OFF pairs ---
    if len(pairs) >= 5:
        X_p, Y_p, feat_cols_p, _ = prepare_pair_dataset(pairs)
        scaler_p = StandardScaler()
        X_p_scaled = scaler_p.fit_transform(X_p)

        for target in Y_p.columns:
            y = Y_p[target].values
            add("-" * 72)
            add(f"ON/OFF PAIR TARGET: {target}")
            add("-" * 72)
            add(f"(n = {len(y)} pairs)")

            # Just use default RF for pairs (too few samples for grid search)
            rf = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2, random_state=42)
            m = evaluate_model(rf, X_p_scaled, y, LeaveOneOut())
            add(f"\nDefault RF LOO:  R²={m['R2']:.4f}, "
                f"RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

            # Internal validation
            kf = kfold_repeated(rf, X_p_scaled, y, n_splits=3)
            add(f"Repeated 3-fold: R² = {kf['R2_mean']:.3f} "
                f"± {kf['R2_std']:.3f}")

            bs = bootstrap_r2(rf, X_p_scaled, y)
            add(f"Bootstrap 95% CI: [{bs['R2_CI_low']:.3f}, "
                f"{bs['R2_CI_high']:.3f}]")
            add("")

    add("=" * 72)
    add("INTERPRETATION GUIDE")
    add("=" * 72)
    add("""
- LOO R² is the point estimate of out-of-sample performance.
- Repeated K-fold R² gives a distribution: if mean is high but std is also
  high, the model is unstable (predictions depend heavily on which samples
  it saw). Trust models with tight std only.
- Bootstrap 95% CI quantifies uncertainty in the R² estimate itself.
  If CI is wide (e.g. [-0.5, 0.5]) the R² value is essentially
  uninformative; if narrow (e.g. [0.6, 0.75]) the performance is robust.
- Hyperparameter tuning rarely lifts R² dramatically on very small
  datasets, but it does reduce variance and produces more stable models.
""")

    report = '\n'.join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'ml_improve_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n\nReport saved to: {path}")


if __name__ == '__main__':
    run_improvement_analysis()
