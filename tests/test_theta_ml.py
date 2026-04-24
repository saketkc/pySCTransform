"""
Tests for theta_ml function comparing against R's sctransform implementation.
"""
import numpy as np
import pytest
import statsmodels.discrete.discrete_model

from pysctransform.fit import theta_ml
from tests.utils import get_pbmc3k_filtered


@pytest.fixture(scope="session")
def pbmc3k_data(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("pbmc3k_cache")
    return get_pbmc3k_filtered(cache_dir)


@pytest.mark.network
class TestThetaML:

    def test_theta_ml_matches_r_implementation(
        self, pbmc3k_with_model, r_reference_data,
    ):
        """
        Verify that theta_ml with defaults matches R's sctransform output.
        """
        matrix, genes, cells, design_matrix = pbmc3k_with_model
        r_raw = r_reference_data

        results = []

        for gene in list(r_raw.index)[:30]:
            if gene not in genes:
                continue

            gene_idx = genes.index(gene)
            gene_umi = np.asarray(matrix[gene_idx, :].todense(), dtype=int).flatten()

            y = np.asarray(gene_umi, dtype=int)
            model = statsmodels.discrete.discrete_model.Poisson(y, design_matrix)

            try:
                fit = model.fit(disp=False)
                mu = fit.predict()
            except Exception:
                continue

            r_theta = r_raw.loc[gene, 'theta']
            py_theta = theta_ml(y, mu)

            # Calculate difference
            if np.isinf(py_theta) and np.isinf(r_theta):
                diff = 0
            elif np.isinf(py_theta) or np.isinf(r_theta):
                diff = np.inf
            else:
                diff = abs(r_theta - py_theta)

            rel_diff = diff / max(r_theta, 0.001) if np.isfinite(diff) else np.inf
            is_match = diff < 0.001 or rel_diff < 0.01

            results.append(
                {
                    'gene': gene,
                    'r_theta': r_theta,
                    'py_theta': py_theta,
                    'diff': diff,
                    'match': is_match
                },
            )

        # Check that we have enough matches
        match_count = sum(r['match'] for r in results)
        total_count = len(results)
        match_rate = match_count / total_count

        assert total_count > 0, "No genes were tested"
        assert match_rate >= 0.95, (
            f"Match rate {match_rate:.1%} ({match_count}/{total_count}) " +
            "is below 95% threshold. "
            f"Failed genes: {[r['gene'] for r in results if not r['match']]}"
        )

    @pytest.mark.parametrize("gene_idx", [0, 10, 50, 100])
    def test_theta_ml_returns_positive(self, pbmc3k_with_model, gene_idx):
        """Verify theta_ml returns positive values for various genes."""
        matrix, genes, cells, design_matrix = pbmc3k_with_model

        if gene_idx >= len(genes):
            pytest.skip(f"Gene index {gene_idx} out of range")

        gene_umi = np.asarray(matrix[gene_idx, :].todense(), dtype=int).flatten()
        y = np.asarray(gene_umi, dtype=int)

        model = statsmodels.discrete.discrete_model.Poisson(y, design_matrix)
        fit = model.fit(disp=False)
        mu = fit.predict()

        theta = theta_ml(y, mu)

        assert theta > 0, f"theta_ml should return positive value, got {theta}"

    def test_theta_ml_with_uniform_mu(self):
        """Test theta_ml with simple uniform mu values."""
        np.random.seed(42)
        y = np.random.negative_binomial(n=5, p=0.5, size=100)
        mu = np.full(100, y.mean())

        theta = theta_ml(y, mu)

        assert np.isfinite(theta) or np.isinf(theta), "theta should be finite or inf"
        assert theta > 0, "theta should be positive"
