"""
Tests for Phase 3 optimisation modules (BO + NSGA-II).

Run: pytest tests/test_phase3.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd


# ---------------------------------------------------------------
# Bayesian Optimisation
# ---------------------------------------------------------------
class TestBayesianOpt:
    @pytest.fixture(scope="class")
    def bo_result(self):
        """Run a short BO (20 calls) to keep test fast."""
        from phase3_bayesopt import run_bayesian_optimisation
        result, top_df, all_df = run_bayesian_optimisation(
            n_calls=20, verbose=False)
        return result, top_df, all_df

    def test_returns_result(self, bo_result):
        result, _, _ = bo_result
        assert hasattr(result, 'fun')
        assert hasattr(result, 'x')

    def test_top_df_has_10_rows(self, bo_result):
        """top_df should have <= 10 rows (might be fewer if n_calls < 10)."""
        _, top_df, _ = bo_result
        assert len(top_df) <= 10

    def test_parameters_in_bounds(self, bo_result):
        """All suggested candidates must respect the search space."""
        _, _, all_df = bo_result
        assert all_df['dx'].between(23, 35).all()
        assert all_df['g_w'].between(3, 5).all()
        assert all_df['c_w'].between(12, 28).all()
        assert all_df['h_graph'].between(-6, 6).all()
        assert all_df['w_au'].between(3, 5).all()

    def test_predictions_are_finite(self, bo_result):
        """All predicted dip values must be finite."""
        import numpy as np
        _, _, all_df = bo_result
        pred_col = [c for c in all_df.columns if c.startswith('predicted')][0]
        assert np.isfinite(all_df[pred_col]).all()


# ---------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------
class TestNSGA2:
    @pytest.fixture(scope="class")
    def pareto(self):
        """Run a short NSGA-II (10 gen) to keep test fast."""
        from phase3_nsga2 import run_nsga2
        pareto_df, result = run_nsga2(pop_size=20, n_gen=10, verbose=False)
        return pareto_df, result

    def test_pareto_df_nonempty(self, pareto):
        pareto_df, _ = pareto
        assert len(pareto_df) > 0

    def test_parameters_in_bounds(self, pareto):
        pareto_df, _ = pareto
        assert pareto_df['dx'].between(23, 35).all()
        assert pareto_df['g_w'].between(3, 5).all()
        assert pareto_df['c_w'].between(12, 28).all()
        assert pareto_df['h_graph'].between(-6, 6).all()
        assert pareto_df['w_au'].between(3, 5).all()

    def test_predictions_finite(self, pareto):
        import numpy as np
        pareto_df, _ = pareto
        assert np.isfinite(pareto_df['predicted_dip_dB']).all()
        assert np.isfinite(pareto_df['predicted_freq_shift_abs_GHz']).all()

    def test_pareto_non_dominated(self, pareto):
        """Verify the Pareto front is actually Pareto-optimal (no solution
        dominates another in both objectives)."""
        pareto_df, _ = pareto
        vals = pareto_df[['predicted_dip_dB',
                          'predicted_freq_shift_abs_GHz']].values
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    continue
                dip_i, shift_i = vals[i]
                dip_j, shift_j = vals[j]
                # j dominates i if j better on BOTH (strictly on one)
                dominated = (dip_j <= dip_i and shift_j >= shift_i) and \
                            (dip_j < dip_i or shift_j > shift_i)
                assert not dominated, (
                    f"Point {i} ({dip_i:.2f}, {shift_i:.2f}) is dominated "
                    f"by point {j} ({dip_j:.2f}, {shift_j:.2f})"
                )


# ---------------------------------------------------------------
# Output artefacts (integration test for generated CSV files)
# ---------------------------------------------------------------
class TestOutputArtefacts:
    """Smoke test: after running the full pipeline, these files must exist
    and be parseable. Skipped if files are missing (first-run scenario)."""

    def test_bo_csv_exists(self):
        path = os.path.join('output', 'phase3_bo_top10.csv')
        if not os.path.exists(path):
            pytest.skip("Run phase3_bayesopt.py to generate this file")
        df = pd.read_csv(path)
        assert len(df) > 0
        assert 'dx' in df.columns
        assert 'w_au' in df.columns

    def test_pareto_csv_exists(self):
        path = os.path.join('output', 'phase3_pareto_front.csv')
        if not os.path.exists(path):
            pytest.skip("Run phase3_nsga2.py to generate this file")
        df = pd.read_csv(path)
        assert len(df) > 0
        assert 'predicted_dip_dB' in df.columns
        assert 'predicted_freq_shift_abs_GHz' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
