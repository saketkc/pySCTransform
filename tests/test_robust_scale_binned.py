import numpy as np
import pytest

from pysctransform.pysctransform import robust_scale_binned


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100_000
    return {
        'y': np.random.randn(n),
        'x': np.random.rand(n),
        'breaks': np.linspace(0, 1, 21),  # 20 bins
    }


class TestRobustScaleBinned:

    def test_preserves_input_order(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        x = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        breaks = [0, 0.5, 1.0]

        result = robust_scale_binned(y, x, breaks)

        # Check that high-x values (indices 0, 2, 4) are scaled together
        # and low-x values (indices 1, 3, 5) are scaled together
        high_bin = result[[0, 2, 4]]
        low_bin = result[[1, 3, 5]]

        # Within each bin, relative ordering should be preserved
        assert high_bin[0] < high_bin[1] < high_bin[2]
        assert low_bin[0] < low_bin[1] < low_bin[2]

    def test_output_length_matches_input(self, sample_data):
        result = robust_scale_binned(
            sample_data['y'],
            sample_data['x'],
            sample_data['breaks'],
        )
        assert len(result) == len(sample_data['y'])

    def test_no_nans_in_output(self, sample_data):
        result = robust_scale_binned(
            sample_data['y'],
            sample_data['x'],
            sample_data['breaks'],
        )
        assert not np.isnan(result).any()

    def test_nan_x_values_remain_nan(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0.2, np.nan, 0.8, np.nan, 0.5])
        breaks = [0, 0.5, 1.0]

        result = robust_scale_binned(y, x, breaks)

        expected_nan = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(np.isnan(result), expected_nan)
        np.testing.assert_array_equal(np.isfinite(result), ~expected_nan)

    def test_empty_bins_do_not_raise(self):
        # All x values in [0.1, 0.4], so bins covering (0.5, 1.0] are empty
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([0.1, 0.2, 0.3, 0.35, 0.4])
        breaks = [0, 0.25, 0.5, 0.75, 1.0]

        result = robust_scale_binned(y, x, breaks)

        assert len(result) == len(y)
        assert np.all(np.isfinite(result))

    @pytest.mark.benchmark
    def test_robust_scale_binned_performance(self, sample_data, benchmark):
        result = benchmark(
            robust_scale_binned,
            sample_data['y'],
            sample_data['x'],
            sample_data['breaks'],
        )
        assert len(result) == len(sample_data['y'])