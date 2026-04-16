"""
Tests for data_loader.py

Run: pytest tests/test_data_loader.py -v
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from data_loader import parse_filename, load_all_data, pair_on_off, find_dip
from config import DATA_DIR


# ---------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------
class TestParseFilename:
    """Test the regex parser for COMSOL simulation filenames."""

    def test_basic_sim(self):
        p = parse_filename("Sdx35_gw3cw28wgraph1hgraph2wau4_sigma0.3.txt")
        assert p['dx'] == 35
        assert p['g_w'] == 3
        assert p['c_w'] == 28
        assert p['w_graph'] == 1
        assert p['h_graph'] == 2
        assert p['w_au'] == 4
        assert p['sigma'] == 0.3

    def test_default_dx(self):
        """Files starting with 'S_' (no dx prefix) should default to dx=35."""
        p = parse_filename("S_gw3cw28wgraph1hgraph2wau4_sigma0.3.txt")
        assert p['dx'] == 35

    def test_negative_hgraph(self):
        """The 'm' prefix means negative (e.g., hgraphm6 -> -6)."""
        p = parse_filename("Sdx35_gw3cw28wgraph1hgraphm6wau4_sigma0.3.txt")
        assert p['h_graph'] == -6

    def test_windows_duplicate_suffix(self):
        """Windows download suffixes '(1)', '(2)' should be stripped."""
        p = parse_filename(
            "Sdx35_gw3cw28wgraph1hgraph2wau4_sigma0.3 (1) (2).txt")
        assert p['dx'] == 35
        assert p['sigma'] == 0.3

    def test_double_underscore(self):
        """Files with '__sigma' should parse fine."""
        p = parse_filename(
            "Sdx35_gw3cw28wgraph1hgraph0wau4__sigma0.3.txt")
        assert p['sigma'] == 0.3

    def test_sigma_1_2(self):
        """OFF state conductivity."""
        p = parse_filename("Sdx23_gw3cw28wgraph1hgraph2wau4_sigma1.2.txt")
        assert p['sigma'] == 1.2

    def test_new_wau5_filename(self):
        """Phase 3 added w_au=5 variation."""
        p = parse_filename(
            "Sdx35_gw3cw28wgraph1hgraphm2wau5_sigma0.3.txt")
        assert p['w_au'] == 5
        assert p['h_graph'] == -2


# ---------------------------------------------------------------
# find_dip
# ---------------------------------------------------------------
class TestFindDip:
    def test_single_dip(self):
        import numpy as np
        freq = np.array([100, 200, 300, 400, 500])
        s = np.array([-1, -5, -20, -3, -2])
        dip_val, dip_freq = find_dip(freq, s)
        assert dip_val == -20
        assert dip_freq == 300

    def test_multiple_dips_picks_deepest(self):
        import numpy as np
        freq = np.array([100, 200, 300, 400, 500])
        s = np.array([-1, -10, -5, -15, -2])
        dip_val, dip_freq = find_dip(freq, s)
        assert dip_val == -15
        assert dip_freq == 400


# ---------------------------------------------------------------
# load_all_data
# ---------------------------------------------------------------
class TestLoadAllData:
    """Integration test against the real data/ folder."""

    @classmethod
    def setup_class(cls):
        cls.df = load_all_data(DATA_DIR)

    def test_row_count(self):
        """After Phase 3, we expect exactly 27 unique simulations."""
        assert len(self.df) == 27, (
            f"Expected 27 simulations after Phase 3 data integration, "
            f"got {len(self.df)}"
        )

    def test_required_columns(self):
        required = ['dx', 'g_w', 'c_w', 'h_graph', 'w_graph', 'w_au',
                    'sigma', 's12_min', 's12_min_freq']
        for col in required:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_no_null_parameters(self):
        params = ['dx', 'g_w', 'c_w', 'h_graph', 'w_graph', 'w_au', 'sigma']
        assert not self.df[params].isnull().any().any(), \
            "Parameter columns should never contain NaN"

    def test_sigma_values(self):
        """Only two sigma values exist: 0.3 (ON) and 1.2 (OFF)."""
        sigmas = set(self.df['sigma'].unique())
        assert sigmas == {0.3, 1.2}, f"Unexpected sigma values: {sigmas}"

    def test_wau_now_varies(self):
        """Phase 3 introduced w_au variance — this must not regress."""
        assert self.df['w_au'].nunique() >= 2, (
            "w_au should have at least 2 distinct values after Phase 3; "
            "if this fails, the zero-variance artefact has returned."
        )

    def test_hgraph_includes_m2(self):
        """Phase 3 added h_graph=-2 configuration."""
        assert -2 in self.df['h_graph'].unique(), \
            "h_graph=-2 should be present (added in Phase 3)"

    def test_s12_dip_is_negative(self):
        """S12 dip in dB should be negative (transmission < 1)."""
        assert (self.df['s12_min'] < 0).all(), \
            "All S12 dip values should be negative"

    def test_frequency_range_reasonable(self):
        """Resonance frequency should be in THz range (250-1000 GHz)."""
        assert (self.df['s12_min_freq'] >= 250).all()
        assert (self.df['s12_min_freq'] <= 1000).all()


# ---------------------------------------------------------------
# pair_on_off
# ---------------------------------------------------------------
class TestPairOnOff:
    @classmethod
    def setup_class(cls):
        cls.df = load_all_data(DATA_DIR)
        cls.pairs = pair_on_off(cls.df)

    def test_pair_count(self):
        """After Phase 3, we expect 13 complete ON/OFF pairs."""
        assert len(self.pairs) == 13, (
            f"Expected 13 ON/OFF pairs after Phase 3, got {len(self.pairs)}"
        )

    def test_pair_count_grew(self):
        """Phase 3 must not have reduced pair count below Phase 2 (7)."""
        assert len(self.pairs) > 7

    def test_freq_shift_computed(self):
        assert 'freq_shift_ghz' in self.pairs.columns
        assert (self.pairs['freq_shift_ghz'] >= 0).all()

    def test_avg_dip_negative(self):
        assert (self.pairs['avg_dip'] < 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
