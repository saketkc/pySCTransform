"""
Test step 3: Pearson residuals calculation.

Uses R's fitted parameters as input to test only the residual calculation step.
"""
import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from pysctransform.pysctransform import (
    get_residuals,
    make_cell_attr,
    row_gmean,
)
from tests.utils import pbmc3k_data


@pytest.fixture(scope="session")
def pbmc3k_filtered_for_residuals(pbmc3k_data):
    """Filter data to genes with min_cells >= 5 (matching vst default)."""
    matrix, genes, cells = pbmc3k_data
    genes = np.array(genes)

    genes_cell_count = np.asarray((matrix >= 0.01).sum(1)).squeeze()
    min_cells_mask = genes_cell_count >= 5
    matrix = matrix[min_cells_mask, :]
    genes = genes[min_cells_mask]

    return matrix, genes, cells


@pytest.fixture(scope="session")
def r_fitted_params():
    """Load R's fitted parameters."""
    r_fit = pd.read_csv("./data/r_model_pars_fit.csv", index_col=0)
    return r_fit[~r_fit.index.duplicated(keep='first')]


@pytest.fixture(scope="session")
def r_residuals():
    """Load R's Pearson residuals (expected output of step 3)."""
    # This file should contain R's Pearson residuals matrix
    # Rows = genes, Columns = cells
    r_res = pd.read_csv("./data/r_residuals.csv", index_col=0)
    return r_res


@pytest.fixture(scope="session")
def residual_inputs(pbmc3k_filtered_for_residuals, r_fitted_params):
    """Prepare all inputs needed for residual calculation."""
    matrix, genes, cells = pbmc3k_filtered_for_residuals

    # Create cell attributes and model matrix
    cell_attr = make_cell_attr(matrix, cells)
    model_matrix = dmatrix("log_umi", cell_attr)

    # Prepare fitted parameters in Python format
    # Only keep genes that are in both our data and R's fitted params
    common_genes = [g for g in genes if g in r_fitted_params.index]
    gene_mask = np.isin(genes, common_genes)

    matrix_filtered = matrix[gene_mask, :]
    genes_filtered = genes[gene_mask]

    # Create model_parameters_fit DataFrame with correct column names
    model_parameters_fit = pd.DataFrame(index=genes_filtered)
    model_parameters_fit['Intercept'] = [
        r_fitted_params.loc[g, '(Intercept)'] for g in genes_filtered
    ]
    model_parameters_fit['log_umi'] = [
        r_fitted_params.loc[g, 'log_umi'] for g in genes_filtered
    ]
    model_parameters_fit['theta'] = [
        r_fitted_params.loc[g, 'theta'] for g in genes_filtered
    ]

    return {
        'matrix': matrix_filtered,
        'genes': genes_filtered,
        'cells': cells,
        'model_matrix': model_matrix,
        'model_parameters_fit': model_parameters_fit,
    }


class TestStep3:
    def test_residuals_calculation(self, residual_inputs, r_residuals):
        """
        Test step 3 (Pearson residuals) using R's fitted parameters as input.

        Verifies that Python's residual calculation matches R's output.
        """
        print(f"\nR residuals shape: {r_residuals.shape}")
        print(
            f"Python input: {len(residual_inputs['genes'])} genes, "
            f"{len(residual_inputs['cells'])} cells"
            )

        # Calculate Python residuals
        print("\nCalculating residuals...")
        py_residuals = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        # Convert to DataFrame for easier comparison
        py_residuals_df = pd.DataFrame(
            py_residuals,
            index=residual_inputs['genes'],
            columns=residual_inputs['cells'],
        )

        print(f"Python residuals shape: {py_residuals_df.shape}")

        # Find common genes and cells
        common_genes = list(
            set(r_residuals.index) & set(py_residuals_df.index)
        )[:100]
        common_cells = list(
            set(r_residuals.columns) & set(py_residuals_df.columns)
        )[:100]

        print(f"Comparing {len(common_genes)} genes x {len(common_cells)} cells")

        # Compare residuals for a sample of genes
        print("\n" + "=" * 80)
        print(
            f"{'Gene':<15} {'R_mean':>10} {'Py_mean':>10} {'R_std':>10} "
            f"{'Py_std':>10} {'Corr':>10} {'OK':>5}"
            )
        print("-" * 80)

        correlations = []
        mean_diffs = []
        std_diffs = []

        for gene in common_genes[:30]:
            r_vals = r_residuals.loc[gene, common_cells].values.astype(float)
            py_vals = py_residuals_df.loc[gene, common_cells].values.astype(float)

            r_mean = np.mean(r_vals)
            py_mean = np.mean(py_vals)
            r_std = np.std(r_vals)
            py_std = np.std(py_vals)

            # Correlation between R and Python residuals for this gene
            corr = np.corrcoef(r_vals, py_vals)[0, 1]
            correlations.append(corr)
            mean_diffs.append(abs(r_mean - py_mean))
            std_diffs.append(abs(r_std - py_std))

            ok = corr > 0.99 and abs(r_mean - py_mean) < 0.1
            status = "✓" if ok else "✗"

            print(
                f"{gene:<15} {r_mean:>10.4f} {py_mean:>10.4f} "
                f"{r_std:>10.4f} {py_std:>10.4f} {corr:>10.4f} {status:>5}"
                )

        print("=" * 80)

        # Summary statistics
        print(f"\nSummary:")
        print(f"  Mean correlation: {np.mean(correlations):.4f}")
        print(f"  Min correlation: {np.min(correlations):.4f}")
        print(f"  Mean abs diff (means): {np.mean(mean_diffs):.4f}")
        print(f"  Mean abs diff (stds): {np.mean(std_diffs):.4f}")

        # Overall correlation (flatten and compare)
        r_flat = r_residuals.loc[common_genes, common_cells].values.flatten()
        py_flat = py_residuals_df.loc[common_genes, common_cells].values.flatten()
        overall_corr = np.corrcoef(r_flat, py_flat)[0, 1]
        print(f"  Overall correlation: {overall_corr:.4f}")

        # Assertions
        assert np.mean(correlations) > 0.95, \
            f"Mean per-gene correlation {np.mean(correlations):.3f} < 0.95"
        assert overall_corr > 0.95, \
            f"Overall correlation {overall_corr:.3f} < 0.95"

        print("\n✓ Test passed!")

    def test_residuals_clipping(self, residual_inputs):
        """Test that residual clipping works correctly."""
        n_cells = residual_inputs['matrix'].shape[1]

        # Test default clipping
        residuals_default = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        expected_clip = np.sqrt(n_cells)
        assert np.max(residuals_default) <= expected_clip + 1e-6, \
            f"Default clipping failed: max={np.max(residuals_default)}, expected<={expected_clip}"
        assert np.min(residuals_default) >= -expected_clip - 1e-6, \
            f"Default clipping failed: min={np.min(residuals_default)}, expected>={-expected_clip}"

        print(
            f"Default clipping (sqrt(n_cells)={expected_clip:.2f}): "
            f"range=[{np.min(residuals_default):.2f}, {np.max(residuals_default):.2f}]"
            )

        # Test Seurat clipping
        residuals_seurat = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="seurat",
        )

        expected_clip_seurat = np.sqrt(n_cells / 30)
        assert np.max(residuals_seurat) <= expected_clip_seurat + 1e-6, \
            f"Seurat clipping failed: max={np.max(residuals_seurat)}"
        assert np.min(residuals_seurat) >= -expected_clip_seurat - 1e-6, \
            f"Seurat clipping failed: min={np.min(residuals_seurat)}"

        print(
            f"Seurat clipping (sqrt(n_cells/30)={expected_clip_seurat:.2f}): "
            f"range=[{np.min(residuals_seurat):.2f}, {np.max(residuals_seurat):.2f}]"
            )

        print("\n✓ Clipping test passed!")

    def test_residuals_statistics(self, residual_inputs):
        """Test that residuals have expected statistical properties."""
        residuals = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        # Pearson residuals should be approximately mean 0 for well-fitted models
        # (though clipping can shift this slightly)
        gene_means = np.mean(residuals, axis=1)
        gene_vars = np.var(residuals, axis=1)

        print(f"\nResidual statistics:")
        print(
            f"  Gene means: min={np.min(gene_means):.4f}, "
            f"max={np.max(gene_means):.4f}, median={np.median(gene_means):.4f}"
            )
        print(
            f"  Gene variances: min={np.min(gene_vars):.4f}, "
            f"max={np.max(gene_vars):.4f}, median={np.median(gene_vars):.4f}"
            )

        # Most genes should have mean close to 0
        assert np.median(np.abs(gene_means)) < 0.5, \
            f"Median absolute gene mean too high: {np.median(np.abs(gene_means)):.4f}"

        print("\n✓ Statistics test passed!")