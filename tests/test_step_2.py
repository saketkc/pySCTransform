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
def r_raw_params():
    """Load R's raw parameters (input to step 2)."""
    r_raw = pd.read_csv("./data/r_model_pars.csv", index_col=0)
    return r_raw[~r_raw.index.duplicated(keep='first')]


@pytest.fixture(scope="session")
def r_fitted_params():
    """Load R's fitted parameters (expected output of step 2)."""
    r_fit = pd.read_csv("./data/r_model_pars_fit.csv", index_col=0)
    return r_fit[~r_fit.index.duplicated(keep='first')]


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
        print(f"\nR fitted parameters: {len(r_fitted_params)} genes")
        print(
            "Genes in step 1 (after outlier removal): " +
            f"{len(regularization_inputs['genes_step1'])}",
        )
        print(f"Outliers removed: {regularization_inputs['n_outliers']}")

        # Run Python's regularization
        print("\nRunning regularization...")
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

        print(f"Python fitted parameters: {len(py_fit)} genes")

        # Compare fitted parameters
        common_genes = list(r_fitted_params.index.intersection(py_fit.index))[:100]
        print(f"\nComparing {len(common_genes)} genes")
        print("=" * 75)
        print(
            f"{'Gene':<15} {'R_int':>12} {'Py_int':>12} "
            f"{'R_theta':>12} {'Py_theta':>12} {'OK':>5}",
        )
        print("-" * 75)

        matches = 0
        int_diffs = []
        theta_diffs = []

        for gene in common_genes[:50]:
            r_int = float(r_fitted_params.loc[gene, '(Intercept)'])
            r_theta = float(r_fitted_params.loc[gene, 'theta'])
            py_int = float(py_fit.loc[gene, 'Intercept'])
            py_theta = float(py_fit.loc[gene, 'theta'])

            int_diff = abs(py_int - r_int)
            int_diffs.append(int_diff)

            # Use relative tolerance for theta
            if np.isinf(r_theta) and np.isinf(py_theta):
                theta_ok = True
                theta_diff = 0
            elif np.isinf(r_theta) or np.isinf(py_theta):
                theta_ok = False
                theta_diff = np.inf
            else:
                theta_diff = abs(py_theta - r_theta) / max(r_theta, 0.001)
                theta_ok = theta_diff < 0.1  # 10% relative tolerance

            theta_diffs.append(theta_diff)

            int_ok = int_diff < 0.1  # Absolute tolerance for intercept
            all_ok = int_ok and theta_ok

            if all_ok:
                matches += 1

            status = "✓" if all_ok else "✗"
            print(
                f"{gene:<15} {r_int:>12.4f} {py_int:>12.4f} "
                f"{r_theta:>12.4f} {py_theta:>12.4f} {status:>5}",
            )

        # Summary statistics
        print("=" * 75)
        print("\nSummary:")
        print(f"  Matched: {matches}/50 ({100 * matches / 50:.1f}%)")
        print(f"  Intercept mean abs diff: {np.mean(int_diffs):.4f}")
        finite_theta_diffs = [d for d in theta_diffs if np.isfinite(d)]
        print(f"  Theta mean rel diff: {np.mean(finite_theta_diffs):.4f}")

        # Calculate correlations
        r_ints = [float(r_fitted_params.loc[g, '(Intercept)']) for g in common_genes]
        py_ints = [float(py_fit.loc[g, 'Intercept']) for g in common_genes]
        int_corr = np.corrcoef(r_ints, py_ints)[0, 1]

        r_thetas = np.array(
            [float(r_fitted_params.loc[g, 'theta']) for g in common_genes],
        )
        py_thetas = np.array([float(py_fit.loc[g, 'theta']) for g in common_genes])
        finite_mask = np.isfinite(r_thetas) & np.isfinite(py_thetas)
        theta_corr = np.corrcoef(r_thetas[finite_mask], py_thetas[finite_mask])[0, 1]

        print(f"  Intercept correlation: {int_corr:.4f}")
        print(f"  Theta correlation: {theta_corr:.4f}")

        # Assertions
        assert int_corr > 0.95, f"Intercept correlation {int_corr:.3f} < 0.95"
        assert theta_corr > 0.90, f"Theta correlation {theta_corr:.3f} < 0.90"
