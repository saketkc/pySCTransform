"""
Minimal test: Compare step 1 (raw parameters) against R.
Tests only the GLM fitting and theta estimation, not regularization.
"""
import numpy as np
import pytest
from patsy import dmatrix

from pysctransform.pysctransform import (
    get_model_params_per_gene,
    make_cell_attr,
)
from tests.utils import (
    compare_params,
    compute_correlation,
)


@pytest.mark.network
class TestStep1:
    def test_step1_50_genes(self, pbmc3k_with_model, r_reference_data):
        """
        Test step 1 (raw parameter estimation) for 50 genes.

        Directly calls get_model_params_per_gene() to test GLM + theta_ml.
        """
        r_genes = list(r_reference_data.index)
        matrix, genes, cells, _ = pbmc3k_with_model

        cell_attr = make_cell_attr(matrix, cells)
        model_matrix = dmatrix("log_umi", cell_attr)

        test_genes = [g for g in r_genes[:50] if g in genes]
        assert len(
            test_genes) > 0, "No overlapping genes between R reference and Python data"

        matches = 0
        mismatched_genes = []
        py_intercepts, r_intercepts = [], []
        py_slopes, r_slopes = [], []
        py_thetas, r_thetas = [], []

        for gene in test_genes:
            gene_idx = genes.index(gene)
            gene_umi = np.asarray(matrix[gene_idx, :].todense()).flatten()

            params = get_model_params_per_gene(
                gene_umi, model_matrix,
                method="theta_ml",
            )

            assert 'Intercept' in params, f"Missing 'Intercept' for gene {gene}"
            assert 'log_umi' in params, f"Missing 'log_umi' for gene {gene}"
            assert 'theta' in params, f"Missing 'theta' for gene {gene}"

            py_int = params['Intercept']
            py_slope = params['log_umi']
            py_theta = params['theta']

            assert np.isfinite(py_int), f"Non-finite intercept for gene {gene}"
            assert np.isfinite(py_slope), f"Non-finite slope for gene {gene}"
            assert py_theta > 0, f"Non-positive theta for gene {gene}: {py_theta}"

            r_int = float(r_reference_data.loc[gene, '(Intercept)'])
            r_slope = float(r_reference_data.loc[gene, 'log_umi'])
            r_theta = float(r_reference_data.loc[gene, 'theta'])

            py_intercepts.append(py_int)
            r_intercepts.append(r_int)
            py_slopes.append(py_slope)
            r_slopes.append(r_slope)
            py_thetas.append(py_theta)
            r_thetas.append(r_theta)

            int_ok, _ = compare_params(py_int, r_int, atol=0.001)
            slope_ok, _ = compare_params(py_slope, r_slope, atol=0.001)
            theta_ok, _ = compare_params(py_theta, r_theta, atol=0.001)

            if int_ok and slope_ok and theta_ok:
                matches += 1
            else:
                mismatched_genes.append(gene)

        match_rate = matches / len(test_genes)
        int_corr = compute_correlation(py_intercepts, r_intercepts)
        slope_corr = compute_correlation(py_slopes, r_slopes)
        theta_corr = compute_correlation(py_thetas, r_thetas)

        assert match_rate >= 0.98, (
            f"Match rate too low: {match_rate:.1%} "
            f"({len(mismatched_genes)} mismatched: {mismatched_genes[:10]})"
        )
        assert int_corr > 0.999, f"Intercept correlation too low: {int_corr:.4f}"
        assert slope_corr > 0.999, f"Slope correlation too low: {slope_corr:.4f}"
        assert theta_corr > 0.99, f"Theta correlation too low: {theta_corr:.4f}"
