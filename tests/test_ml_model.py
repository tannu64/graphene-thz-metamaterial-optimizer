"""
Tests for ml_model.py — verifies model training, CV metrics, feature importance.

Run: pytest tests/test_ml_model.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from config import DATA_DIR
from data_loader import load_all_data, pair_on_off
from ml_model import (
    prepare_dataset, prepare_pair_dataset, build_models,
    evaluate_models, train_final_models, feature_importance,
)


# ---------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset():
    df = load_all_data(DATA_DIR)
    pairs = pair_on_off(df)
    return df, pairs


@pytest.fixture(scope="module")
def all_sim_data(dataset):
    df, _ = dataset
    X, Y, feat_cols, target_cols = prepare_dataset(df)
    return X, Y, feat_cols, target_cols


@pytest.fixture(scope="module")
def pair_data(dataset):
    _, pairs = dataset
    X, Y, feat_cols, target_cols = prepare_pair_dataset(pairs)
    return X, Y, feat_cols, target_cols


# ---------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------
class TestPrepareDataset:
    def test_drops_w_graph(self, all_sim_data):
        X, Y, feat_cols, _ = all_sim_data
        assert 'w_graph' not in feat_cols, \
            "w_graph has zero variance and should be dropped by default"

    def test_wau_retained(self, all_sim_data):
        """w_au must survive feature selection (now has variance)."""
        _, _, feat_cols, _ = all_sim_data
        assert 'w_au' in feat_cols

    def test_target_column_names(self, all_sim_data):
        _, Y, _, target_cols = all_sim_data
        assert target_cols == ['S12 Dip (dB)', 'Dip Freq (GHz)']

    def test_shape_alignment(self, all_sim_data):
        X, Y, _, _ = all_sim_data
        assert len(X) == len(Y)
        assert len(X) == 27


# ---------------------------------------------------------------
# evaluate_models
# ---------------------------------------------------------------
class TestEvaluateModels:
    @pytest.fixture(scope="class")
    def results(self, all_sim_data):
        X, Y, _, _ = all_sim_data
        models = build_models()
        results, _ = evaluate_models(X, Y, models)
        return results

    def test_all_models_evaluated(self, results):
        expected = {'Random Forest', 'Gradient Boosting', 'Gaussian Process'}
        assert set(results.keys()) == expected

    def test_all_targets_have_metrics(self, results):
        for model, targets in results.items():
            for target, metrics in targets.items():
                assert 'R2' in metrics
                assert 'RMSE' in metrics
                assert 'MAE' in metrics
                assert 'y_true' in metrics
                assert 'y_pred' in metrics

    def test_rmse_positive(self, results):
        for model, targets in results.items():
            for target, metrics in targets.items():
                assert metrics['RMSE'] > 0

    def test_mae_positive(self, results):
        for model, targets in results.items():
            for target, metrics in targets.items():
                assert metrics['MAE'] > 0

    def test_freq_prediction_reasonable(self, results):
        """Dip frequency prediction should have R² > 0.3 for at least one model."""
        best = max(
            r['Dip Freq (GHz)']['R2']
            for r in results.values()
        )
        assert best > 0.3, (
            f"Best frequency R² = {best:.3f} is suspiciously low. "
            f"Phase 3 expected ~0.68; regression indicator."
        )


# ---------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------
class TestFeatureImportance:
    @pytest.fixture(scope="class")
    def imp(self, all_sim_data):
        X, Y, feat_cols, _ = all_sim_data
        return feature_importance(X, Y, feat_cols)

    def test_importance_sums_close_to_1(self, imp):
        """RF importances should sum to ~1.0 per target."""
        for col in imp.columns:
            total = imp[col].sum()
            assert 0.95 <= total <= 1.05, (
                f"Feature importance for '{col}' sums to {total}, "
                f"expected ~1.0"
            )

    def test_wau_nonzero(self, imp):
        """w_au must have non-zero importance (Phase 3 regression check)."""
        assert imp.loc['w_au'].max() > 0.001, (
            "w_au has ~zero importance; this was a Phase 2 artefact that "
            "Phase 3 resolved. If it returns, the dataset has regressed."
        )

    def test_cw_dominates_frequency(self, imp):
        """Physics check: c_w should be most important for frequency."""
        freq_col = 'Dip Freq (GHz)'
        top_feature = imp[freq_col].idxmax()
        assert top_feature in ('c_w', 'dx'), (
            f"Expected c_w or dx to dominate frequency prediction, "
            f"got {top_feature}. Check the physics interpretation."
        )


# ---------------------------------------------------------------
# train_final_models
# ---------------------------------------------------------------
class TestTrainFinalModels:
    def test_trained_dict_structure(self, all_sim_data):
        X, Y, _, _ = all_sim_data
        trained, scaler = train_final_models(X, Y, build_models())
        assert set(trained.keys()) == {
            'Random Forest', 'Gradient Boosting', 'Gaussian Process',
        }
        for model_name, targets in trained.items():
            assert set(targets.keys()) == {
                'S12 Dip (dB)', 'Dip Freq (GHz)',
            }

    def test_prediction_works(self, all_sim_data):
        X, Y, _, _ = all_sim_data
        trained, scaler = train_final_models(X, Y, build_models())
        # Predict on the first training row (should produce a finite number)
        x = X.iloc[:1].values
        pred = trained['Gaussian Process']['Dip Freq (GHz)'].predict(
            scaler.transform(x))[0]
        assert np.isfinite(pred)
        # Frequency prediction should be in the simulated range
        assert 250 < pred < 1000


# ---------------------------------------------------------------
# Pair data
# ---------------------------------------------------------------
class TestPairData:
    def test_pair_count(self, pair_data):
        X, Y, _, _ = pair_data
        assert len(X) == 13, "Expected 13 ON/OFF pairs after Phase 3"

    def test_pair_targets(self, pair_data):
        _, _, _, target_cols = pair_data
        assert target_cols == ['Freq Shift (GHz)', 'Avg Dip (dB)']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
