import numpy as np
import pandas as pd
import pytest

from pysctransform.pysctransform import (
    get_regularized_params,
    is_outlier,
    row_gmean,
    vst,
)
from tests.utils import (
    build_batch_column_map,
    compare_gene_parameter,
)


@pytest.mark.network
class TestStep2Batch:
    def test_batch_regularization_50_genes(
        self, pbmc3k_batch_model, r_batch_fit_reference,
    ):
        """
        Test step 2 (regularized parameters) with batch variable for 50 genes.
        End-to-end through vst, checking theta correlation.
        """
        test_genes = list(r_batch_fit_reference.index[:50])

        matrix, genes, cells, cell_attr, design_matrix, formula = pbmc3k_batch_model

        vst_out = vst(
            matrix,
            gene_names=genes,
            cell_names=cells,
            cell_attr_extra=cell_attr[["batch"]],
            batch_var="batch",
            n_genes=None,
            n_cells=None,
            method="theta_ml",
            min_cells=10,
            verbosity=False,
        )

        py_fit = vst_out["model_parameters_fit"]
        _, theta_corr, tested = compare_gene_parameter(
            py_fit, r_batch_fit_reference, test_genes, param="theta", rtol=0.1,
        )

        assert tested > 0, "No genes could be compared"
        assert theta_corr > 0.90, f"Theta correlation too low: {theta_corr:.4f}"

    def test_batch_regularization_with_r_inputs(
        self, pbmc3k_batch_model, r_batch_reference, r_batch_fit_reference,
    ):
        """
        Test step 2 only, using R's raw parameters as input.
        Isolates regularization from step 1 differences.
        """
        r_fit = r_batch_fit_reference

        matrix, genes, cells, cell_attr, design_matrix, formula = pbmc3k_batch_model

        genes_arr = np.array(genes)
        genes_cell_count = np.asarray((matrix >= 0.01).sum(axis=1)).squeeze()
        min_cells_mask = genes_cell_count >= 10
        filtered_indices = np.where(min_cells_mask)[0]
        genes_filtered = genes_arr[min_cells_mask]

        unique_mask = ~pd.Series(genes_filtered).duplicated().values
        genes_filtered = genes_filtered[unique_mask]
        filtered_indices = filtered_indices[unique_mask]
        matrix_filtered = matrix[filtered_indices, :]

        genes_log10_gmean = np.log10(row_gmean(matrix_filtered, gmean_eps=1))

        common_genes = [g for g in r_batch_reference.index if g in genes_filtered]
        assert len(common_genes) > 0, \
            "No common genes between R raw params and filtered data"
        genes_step1 = np.array(common_genes)

        genes_step1_indices = np.array(
            [np.where(genes_filtered == g)[0][0] for g in genes_step1],
        )
        genes_log10_gmean_step1 = genes_log10_gmean[genes_step1_indices]

        model_parameters = r_batch_reference.loc[genes_step1].copy()
        col_map = build_batch_column_map(model_parameters.columns)
        model_parameters = model_parameters.rename(columns=col_map)
        model_parameters['od_factor'] = np.log10(
            1 + np.power(10, genes_log10_gmean_step1)
            / model_parameters['theta'].values,
        )

        outliers_df = pd.DataFrame(index=genes_step1)
        for col in model_parameters.columns:
            outliers_df[col] = is_outlier(
                model_parameters[col].values,
                genes_log10_gmean_step1,
            )

        non_outliers = outliers_df.sum(axis=1) == 0
        genes_step1_clean = genes_step1[non_outliers.values]
        genes_log10_gmean_step1_clean = genes_log10_gmean_step1[non_outliers.values]
        model_parameters_clean = model_parameters.loc[genes_step1_clean]

        py_fit = get_regularized_params(
            model_parameters=model_parameters_clean,
            genes=genes_filtered,
            genes_step1=genes_step1_clean,
            genes_log10_gmean_step1=genes_log10_gmean_step1_clean,
            genes_log10_gmean=genes_log10_gmean,
            cell_attr=cell_attr,
            umi=matrix_filtered,
            batch_var="batch",
            theta_regularization="od_factor",
        )

        assert len(py_fit) > 0, "Regularization produced no results"

        test_genes = list(r_fit.index[:50])

        _, theta_corr, tested = compare_gene_parameter(
            py_fit, r_fit, test_genes, param="theta", rtol=0.1,
        )

        assert tested > 0, "No genes could be compared"
        assert theta_corr > 0.90, f"Theta correlation too low: {theta_corr:.4f}"

        common = [g for g in test_genes if g in py_fit.index and g in r_fit.index]
        if len(common) > 10:
            r_thetas = np.array([float(r_fit.loc[g, 'theta']) for g in common])
            py_thetas = np.array([float(py_fit.loc[g, 'theta']) for g in common])
            finite = np.isfinite(r_thetas) & np.isfinite(py_thetas)
            if finite.sum() > 10:
                mean_rel_diff = np.mean(
                    np.abs(r_thetas[finite] - py_thetas[finite])
                    / np.maximum(r_thetas[finite], 0.001),
                )
                assert mean_rel_diff < 0.5, (
                    f"Theta mean relative difference too large: {mean_rel_diff:.4f}"
                )

    def test_regularization_improves_theta_correlation(
            self, pbmc3k_batch_model, r_batch_reference, r_batch_fit_reference
    ):
        """
        Regularization should bring theta closer to R's regularized reference,
        not further away. Raw Step 1 correlates at ~0.97 with R's raw params.
        After regularization, Python fit should correlate well with R's fit.
        If the bug is present (wrong exog in kernel smoother), correlation drops.
        """
        matrix, genes, cells, cell_attr, design_matrix, formula = pbmc3k_batch_model

        vst_out = vst(
            matrix,
            gene_names=genes,
            cell_names=cells,
            cell_attr_extra=cell_attr[["batch"]],
            batch_var="batch",
            n_genes=None,
            n_cells=None,
            method="theta_ml",
            min_cells=10,
            verbosity=False,
        )

        py_raw = vst_out["model_parameters"]
        py_fit = vst_out["model_parameters_fit"]
        r_fit = r_batch_fit_reference

        common_raw = [g for g in r_batch_reference.index if g in py_raw.index]
        py_theta_raw = py_raw.loc[common_raw, "theta"].values.astype(float)
        r_theta_raw = r_batch_reference.loc[common_raw, "theta"].values.astype(float)
        finite_raw = np.isfinite(py_theta_raw) & np.isfinite(r_theta_raw)
        corr_raw = np.corrcoef(py_theta_raw[finite_raw], r_theta_raw[finite_raw])[0, 1]

        common_fit = [g for g in r_fit.index if g in py_fit.index]
        py_theta_fit = py_fit.loc[common_fit, "theta"].values.astype(float)
        r_theta_fit = r_fit.loc[common_fit, "theta"].values.astype(float)
        finite_fit = np.isfinite(py_theta_fit) & np.isfinite(r_theta_fit)
        corr_fit = np.corrcoef(py_theta_fit[finite_fit], r_theta_fit[finite_fit])[0, 1]

        print(f"\nStep 1 raw theta corr vs R raw: {corr_raw:.4f}")
        print(f"Step 2 fit theta corr vs R fit: {corr_fit:.4f}")

        # Step 1 should be high — if not, a different bug exists
        assert corr_raw > 0.90, f"Step 1 raw theta corr too low: {corr_raw:.4f}"

        # Regularization should not destroy the correlation
        assert corr_fit > 0.90, f"Step 2 fit theta corr too low: {corr_fit:.4f}"