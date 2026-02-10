"""
Test: Compare batch step 1 (raw parameters) against R reference.
Tests GLM fitting with batch interaction terms.
"""
import numpy as np
import pytest

from pysctransform.pysctransform import (
    get_model_params_per_gene,
)
from tests.utils import (
    compare_params,
    compute_correlation,
    print_comparison_footer,
)


@pytest.mark.network
class TestStep1Batch:
    def test_design_matrix_shape(self, pbmc3k_batch_model):
        """Verify design matrix has correct structure for 2 batches."""
        _, _, _, _, design_matrix, _ = pbmc3k_batch_model
        columns = design_matrix.design_info.column_names

        # 2 batches: 2 intercepts + 2 slopes = 4 columns, no global intercept
        assert "Intercept" not in columns
        assert len(columns) == 4

    def test_batch_step1_50_genes(self, pbmc3k_batch_model, r_batch_reference):
        """
        Test step 1 (raw parameter estimation) with batch variable for 50 genes.
        """
        r_raw = r_batch_reference
        r_genes = list(r_raw.index)
        r_columns = list(r_raw.columns)
        print(f"R raw parameters: {len(r_raw)} genes")
        print(f"R columns: {r_columns}")

        matrix, genes, cells, cell_attr, design_matrix, formula = pbmc3k_batch_model
        py_columns = design_matrix.design_info.column_names
        print(f"Python design matrix columns: {py_columns}")

        test_genes = [g for g in r_genes[:50] if g in genes]
        print(f"\nTesting {len(test_genes)} genes")

        # Build column mapping between R and Python names
        # R uses e.g. "batchA:log_umi", Python uses e.g. "C(batch)[A]:log10_umi"
        # We'll compare by matching batch level and parameter type
        r_coef_cols = [c for c in r_columns if c != "theta"]
        py_coef_cols = [c for c in py_columns]

        print(f"R coefficient columns: {r_coef_cols}")
        print(f"Python coefficient columns: {py_coef_cols}")
        assert len(r_coef_cols) == len(py_coef_cols), (
            f"Column count mismatch: R has {len(r_coef_cols)}, "
            f"Python has {len(py_coef_cols)}"
        )

        # Match columns by sorting both (they should correspond after sorting)
        # This is fragile — inspect printed columns above to build a proper mapping
        col_map = dict(zip(sorted(py_coef_cols), sorted(r_coef_cols)))
        print(f"Column mapping (Python -> R): {col_map}")

        matches = 0
        py_thetas, r_thetas = [], []

        for gene in test_genes:
            gene_idx = genes.index(gene)
            gene_umi = np.asarray(matrix[gene_idx, :].todense()).flatten()

            params = get_model_params_per_gene(
                gene_umi, design_matrix,
                method="theta_ml",
            )

            py_theta = params["theta"]
            r_theta = float(r_raw.loc[gene, "theta"])

            py_thetas.append(py_theta)
            r_thetas.append(r_theta)

            theta_ok, _ = compare_params(py_theta, r_theta, atol=0.01)

            coef_ok = True
            for py_col, r_col in col_map.items():
                py_val = params[py_col]
                r_val = float(r_raw.loc[gene, r_col])
                ok, _ = compare_params(py_val, r_val, atol=0.001)
                if not ok:
                    coef_ok = False

            all_ok = theta_ok and coef_ok
            if all_ok:
                matches += 1

            status = "✓" if all_ok else "✗"
            print(
                f"{gene:<15} "
                f"R_theta={r_theta:>10.4f} Py_theta={py_theta:>10.4f} "
                f"{status:>5}",
            )

        print_comparison_footer()
        match_rate = matches / len(test_genes)
        theta_corr = compute_correlation(py_thetas, r_thetas)

        print(f"Match rate: {match_rate:.1%}")
        print(f"Theta correlation: {theta_corr:.4f}")

        assert match_rate >= 0.90, f"Match rate too low: {match_rate:.1%}"
        assert theta_corr > 0.99, f"Theta correlation too low: {theta_corr:.4f}"
