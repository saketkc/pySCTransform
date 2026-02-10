"""
Test step 2: Regularization (kernel smoothing of raw parameters).

Uses R's raw parameters as input to test only the regularization step.
"""
import numpy as np
import pandas as pd
import pytest

from pysctransform.pysctransform import (
    get_regularized_params,
    is_outlier,
    row_gmean,
)


@pytest.fixture(scope="session")
def pbmc3k_filtered_for_regularization(pbmc3k_data):
    """Filter data to genes with min_cells >= 5 (matching vst default)."""
    matrix, genes, cells = pbmc3k_data
    genes = np.array(genes)

    genes_cell_count = np.asarray((matrix >= 0.01).sum(1)).squeeze()
    min_cells_mask = genes_cell_count >= 5
    matrix = matrix[min_cells_mask, :]
    genes = genes[min_cells_mask]

    return matrix, genes, cells


@pytest.fixture(scope="session")
def regularization_inputs(pbmc3k_filtered_for_regularization, r_raw_params):
    """Prepare all inputs needed for regularization."""
    matrix, genes, cells = pbmc3k_filtered_for_regularization

    # Calculate gene geometric means
    gmean_eps = 1
    genes_log10_gmean = np.log10(row_gmean(matrix, gmean_eps=gmean_eps))

    # Find common genes between R's raw parameters and our filtered genes
    common_genes = [g for g in r_raw_params.index if g in genes]
    genes_step1 = np.array(common_genes)

    # Get indices preserving genes_step1 order
    genes_step1_indices = np.array(
        [np.where(genes == g)[0][0] for g in genes_step1],
    )
    genes_log10_gmean_step1 = genes_log10_gmean[genes_step1_indices]

    # Prepare model parameters
    model_parameters = r_raw_params.loc[genes_step1].copy()
    model_parameters['od_factor'] = np.log10(
        1 + np.power(10, genes_log10_gmean_step1) / model_parameters['theta'].values,
    )
    model_parameters = model_parameters.rename(columns={'(Intercept)': 'Intercept'})

    # Identify and remove outliers
    outliers_df = pd.DataFrame(index=genes_step1)
    for col in model_parameters.columns:
        outliers_df[col] = is_outlier(
            model_parameters[col].values,
            genes_log10_gmean_step1,
        )

    non_outliers = outliers_df.sum(axis=1) == 0
    genes_step1_clean = genes_step1[non_outliers]
    genes_log10_gmean_step1_clean = genes_log10_gmean_step1[non_outliers]
    model_parameters_clean = model_parameters.loc[genes_step1_clean]

    # Cell attributes
    cell_attr = pd.DataFrame({'umi': np.asarray(matrix.sum(0)).flatten()})

    return {
        'model_parameters': model_parameters_clean,
        'genes': genes,
        'genes_step1': genes_step1_clean,
        'genes_log10_gmean_step1': genes_log10_gmean_step1_clean,
        'genes_log10_gmean': genes_log10_gmean,
        'cell_attr': cell_attr,
        'matrix': matrix,
        'n_outliers': (~non_outliers).sum(),
    }


@pytest.mark.network
class TestStep2:
    def test_regularization_correlation(
        self, regularization_inputs, r_fitted_params,
    ):
        """
        Test step 2 (regularization) using R's raw parameters as input.

        Verifies that Python's regularization produces parameters highly
        correlated with R's fitted parameters.
        """
        py_fit = get_regularized_params(
            model_parameters=regularization_inputs['model_parameters'],
            genes=regularization_inputs['genes'],
            genes_step1=regularization_inputs['genes_step1'],
            genes_log10_gmean_step1=regularization_inputs['genes_log10_gmean_step1'],
            genes_log10_gmean=regularization_inputs['genes_log10_gmean'],
            cell_attr=regularization_inputs['cell_attr'],
            umi=regularization_inputs['matrix'],
            theta_regularization="od_factor",
        )

        assert len(py_fit) > 0, "Regularization produced no results"

        common_genes = list(r_fitted_params.index.intersection(py_fit.index))[:100]
        assert len(
            common_genes) > 0, "No overlapping genes between R fitted and Python fitted"

        # Intercept correlation
        r_ints = np.array(
            [float(r_fitted_params.loc[g, '(Intercept)']) for g in common_genes])
        py_ints = np.array([float(py_fit.loc[g, 'Intercept']) for g in common_genes])
        int_corr = np.corrcoef(r_ints, py_ints)[0, 1]
        int_mean_diff = np.mean(np.abs(r_ints - py_ints))

        # Theta correlation (finite values only)
        r_thetas = np.array(
            [float(r_fitted_params.loc[g, 'theta']) for g in common_genes])
        py_thetas = np.array([float(py_fit.loc[g, 'theta']) for g in common_genes])
        finite_mask = np.isfinite(r_thetas) & np.isfinite(py_thetas)
        assert finite_mask.sum() > 10, (
            f"Too few finite theta values for comparison: {finite_mask.sum()}"
        )
        theta_corr = np.corrcoef(r_thetas[finite_mask], py_thetas[finite_mask])[0, 1]

        assert int_corr > 0.95, (
            f"Intercept correlation {int_corr:.3f} < 0.95 "
            f"(mean abs diff: {int_mean_diff:.4f})"
        )
        assert theta_corr > 0.90, f"Theta correlation {theta_corr:.3f} < 0.90"
        assert int_mean_diff < 0.5, (
            f"Intercept mean absolute difference too large: {int_mean_diff:.4f}"
        )
