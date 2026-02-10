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
)


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


@pytest.mark.network
class TestStep3:
    def test_residuals_calculation(self, residual_inputs, r_residuals):
        """
        Test step 3 (Pearson residuals) using R's fitted parameters as input.

        Verifies that Python's residual calculation matches R's output.
        """
        py_residuals = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        py_residuals_df = pd.DataFrame(
            py_residuals,
            index=residual_inputs['genes'],
            columns=residual_inputs['cells'],
        )

        # Find common genes and cells - convert to SORTED LISTS for consistency
        common_genes = sorted(
            set(r_residuals.index) & set(py_residuals_df.index),
        )[:100]
        common_cells = sorted(
            set(r_residuals.columns) & set(py_residuals_df.columns),
        )[:100]

        assert len(common_genes) > 0, "No overlapping genes"
        assert len(common_cells) > 0, "No overlapping cells"

        # Per-gene correlations
        correlations = []
        low_corr_genes = []

        for gene in common_genes[:30]:
            r_vals = r_residuals.loc[gene, common_cells].values.astype(float)
            py_vals = py_residuals_df.loc[gene, common_cells].values.astype(float)

            corr = np.corrcoef(r_vals, py_vals)[0, 1]
            correlations.append(corr)
            if corr <= 0.99:
                low_corr_genes.append((gene, corr))

        assert np.mean(correlations) > 0.95, (
            f"Mean per-gene correlation {np.mean(correlations):.3f} < 0.95 "
            f"(low correlation genes: {low_corr_genes[:5]})"
        )

        # Overall correlation across all compared values
        r_subset = r_residuals.loc[common_genes, common_cells]
        py_subset = py_residuals_df.loc[common_genes, common_cells]
        assert r_subset.shape == py_subset.shape, (
            f"Shape mismatch: R={r_subset.shape}, Py={py_subset.shape}"
        )

        overall_corr = np.corrcoef(
            r_subset.values.flatten(), py_subset.values.flatten(),
        )[0, 1]

        assert overall_corr > 0.95, (
            f"Overall correlation {overall_corr:.3f} < 0.95"
        )

    def test_residuals_clipping(self, residual_inputs):
        """Test that residual clipping works correctly."""
        n_cells = residual_inputs['matrix'].shape[1]

        # Default clipping: sqrt(n_cells)
        residuals_default = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        expected_clip = np.sqrt(n_cells)
        assert np.max(residuals_default) <= expected_clip + 1e-6, (
            f"Default clipping failed: max={np.max(residuals_default):.4f}, "
            f"expected<={expected_clip:.4f}"
        )
        assert np.min(residuals_default) >= -expected_clip - 1e-6, (
            f"Default clipping failed: min={np.min(residuals_default):.4f}, "
            f"expected>={-expected_clip:.4f}"
        )

        # Seurat clipping: sqrt(n_cells / 30)
        residuals_seurat = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="seurat",
        )

        expected_clip_seurat = np.sqrt(n_cells / 30)
        assert np.max(residuals_seurat) <= expected_clip_seurat + 1e-6, (
            f"Seurat clipping failed: max={np.max(residuals_seurat):.4f}, "
            f"expected<={expected_clip_seurat:.4f}"
        )
        assert np.min(residuals_seurat) >= -expected_clip_seurat - 1e-6, (
            f"Seurat clipping failed: min={np.min(residuals_seurat):.4f}, "
            f"expected>={-expected_clip_seurat:.4f}"
        )

        assert expected_clip_seurat < expected_clip, \
            "Clip range should be tighter than default"

    def test_residuals_statistics(self, residual_inputs):
        """Test that residuals have expected statistical properties."""
        residuals = get_residuals(
            umi=residual_inputs['matrix'],
            model_matrix=residual_inputs['model_matrix'],
            model_parameters_fit=residual_inputs['model_parameters_fit'],
            residual_type="pearson",
            res_clip_range="default",
        )

        gene_means = np.mean(residuals, axis=1)
        gene_vars = np.var(residuals, axis=1)

        # Most genes should have mean close to 0 for well-fitted models
        median_abs_mean = np.median(np.abs(gene_means))
        assert median_abs_mean < 0.5, (
            f"Median absolute gene mean too high: {median_abs_mean:.4f} "
            f"(gene mean range: [{np.min(gene_means):.4f}, {np.max(gene_means):.4f}])"
        )

        # Variances should be positive and finite
        assert np.all(np.isfinite(gene_vars)), "Non-finite gene variances found"
        assert np.all(gene_vars >= 0), "Negative gene variances found"
