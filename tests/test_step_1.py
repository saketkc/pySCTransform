"""
Minimal test: Compare step 1 (raw parameters) against R.
Tests only the GLM fitting and theta estimation, not regularization.
"""
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from pysctransform.pysctransform import (
    make_cell_attr,
    get_model_params_pergene,
)
from tests.utils import get_pbmc3k_filtered


@pytest.fixture(scope="session")
def pbmc3k_data(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("pbmc3k_cache")
    return get_pbmc3k_filtered(cache_dir)

@pytest.fixture(scope="session")
def pbmc3k_with_model(pbmc3k_data):
    matrix, genes, cells = pbmc3k_data

    total_umi_per_cell = np.asarray(matrix.sum(axis=0)).flatten()
    log_umi = np.log10(total_umi_per_cell)

    cell_attr = pd.DataFrame({'log_umi': log_umi}, index=cells)
    design_matrix = dmatrix("log_umi", cell_attr)

    return matrix, genes, cells, design_matrix

@pytest.fixture(scope="session")
def r_reference_data():
    return pd.read_csv("./data/r_model_pars.csv", index_col=0)

class TestStep1:
    def test_step1_50_genes(
            self, pbmc3k_with_model, r_reference_data,
    ):
        """
        Test step 1 (raw parameter estimation) for 50 genes.

        Directly calls get_model_params_pergene() to test GLM + theta_ml.
        """
        # Step 1: Load R reference
        r_raw = r_reference_data
        r_raw = r_raw[~r_raw.index.duplicated(keep='first')]
        r_genes = list(r_raw.index)
        print(f"R raw parameters: {len(r_raw)} genes")

        # Step 2: Load data
        matrix, genes, cells, design_matrix = pbmc3k_with_model
        print(f"Data: {len(genes)} genes, {len(cells)} cells")

        # Step 3: Create cell attributes and model matrix (matching R)
        cell_attr = make_cell_attr(matrix, cells)
        model_matrix = dmatrix("log_umi", cell_attr)

        # Step 4: Select 50 genes that are in R's output
        test_genes = [g for g in r_genes[:50] if g in genes]
        print(f"\nTesting {len(test_genes)} genes")
        print("=" * 85)
        print(
            f"{'Gene':<15} {'R_int':>10} {'Py_int':>10} {'R_slope':>10} {'Py_slope':>10} {'R_theta':>10} {'Py_theta':>10} {'OK':>5}",
        )
        print("-" * 85)

        # Step 5: Fit each gene and compare
        matches = 0
        for gene in test_genes:
            # Get gene data
            gene_idx = genes.index(gene)
            gene_umi = np.asarray(matrix[gene_idx, :].todense()).flatten()

            # Fit using pysctransform's function
            params = get_model_params_pergene(gene_umi, model_matrix, method="theta_ml")

            # Get Python values
            py_int = params['Intercept']
            py_slope = params['log_umi']
            py_theta = params['theta']

            # Get R values
            r_int = float(r_raw.loc[gene, '(Intercept)'])
            r_slope = float(r_raw.loc[gene, 'log_umi'])
            r_theta = float(r_raw.loc[gene, 'theta'])

            # Check match
            int_ok = abs(py_int - r_int) < 0.001
            slope_ok = abs(py_slope - r_slope) < 0.001

            if np.isinf(r_theta) and np.isinf(py_theta):
                theta_ok = True
            elif np.isinf(r_theta) or np.isinf(py_theta):
                theta_ok = False
            else:
                theta_ok = abs(py_theta - r_theta) < 0.001

            all_ok = int_ok and slope_ok and theta_ok
            if all_ok:
                matches += 1

            status = "✓" if all_ok else "✗"
            print(
                f"{gene:<15} {r_int:>10.4f} {py_int:>10.4f} {r_slope:>10.4f} {py_slope:>10.4f} {r_theta:>10.4f} {py_theta:>10.4f} {status:>5}",
            )

        # Step 6: Report and assert
        print("=" * 85)
        match_rate = matches / len(test_genes)
        print(f"Matched: {matches}/{len(test_genes)} ({100 * match_rate:.1f}%)")

        assert match_rate >= 0.98, f"Match rate too low: {match_rate:.1%}"
        print("\n✓ Test passed!")
