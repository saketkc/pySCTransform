import numpy as np
import pandas as pd

from pysctransform.pysctransform import (
    is_outlier,
    vst,
    get_regularized_params,
    row_gmean
)
from tests.utils import (
    build_batch_column_map,
    compare_gene_parameter,
)


class TestStep2Batch:
    def test_batch_regularization_50_genes(
        self, pbmc3k_batch_model, r_batch_reference, r_batch_fit_reference,
    ):
        """
        Test step 2 (regularized parameters) with batch variable for 50 genes.
        """
        r_fit = r_batch_fit_reference
        r_genes = list(r_fit.index)
        test_genes = r_genes[:50]

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
            verbosity=True,
        )

        py_fit = vst_out["model_parameters_fit"]
        match_rate, theta_corr, tested = compare_gene_parameter(
            py_fit, r_fit, test_genes, param="theta", rtol=0.1,
        )
        print(f"Tested: {tested} genes")
        print(f"\nMatch rate: {match_rate:.1%}")
        print(f"Theta correlation: {theta_corr:.4f}")
        assert theta_corr > 0.90, f"Theta correlation too low: {theta_corr:.4f}"

    def test_batch_regularization_with_r_inputs(
        self, pbmc3k_batch_model, r_batch_reference, r_batch_fit_reference,
    ):
        """
        Test step 2 only, using R's raw parameters as input.
        Isolates regularization from step 1 differences.
        """
        r_raw = r_batch_reference
        r_fit = r_batch_fit_reference

        matrix, genes, cells, cell_attr, design_matrix, formula = pbmc3k_batch_model

        # Filter genes same as vst does
        genes_arr = np.array(genes)
        genes_cell_count = np.asarray((matrix >= 0.01).sum(1)).squeeze()
        min_cells_mask = genes_cell_count >= 10
        filtered_indices = np.where(min_cells_mask)[0]
        genes_filtered = genes_arr[min_cells_mask]

        # Remove duplicates
        unique_mask = ~pd.Series(genes_filtered).duplicated().values
        genes_filtered = genes_filtered[unique_mask]
        filtered_indices = filtered_indices[unique_mask]
        matrix_filtered = matrix[filtered_indices, :]

        genes_log10_gmean = np.log10(
            row_gmean(matrix_filtered, gmean_eps=1)
        )

        # Use R's raw parameters
        common_genes = [g for g in r_raw.index if g in genes_filtered]
        genes_step1 = np.array(common_genes)

        genes_step1_indices = np.array(
            [np.where(genes_filtered == g)[0][0] for g in genes_step1],
        )
        genes_log10_gmean_step1 = genes_log10_gmean[genes_step1_indices]

        # Prepare model parameters from R's raw output
        model_parameters = r_raw.loc[genes_step1].copy()

        # Rename R columns to match Python column names\
        col_map = build_batch_column_map(model_parameters.columns)
        model_parameters = model_parameters.rename(columns=col_map)

        # Compute od_factor
        model_parameters['od_factor'] = np.log10(
            1 + np.power(10, genes_log10_gmean_step1) / model_parameters[
                'theta'].values,
        )

        # Remove outliers
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

        # Run regularization
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

        test_genes = list(r_fit.index[:50])
        match_rate, theta_corr, _ = compare_gene_parameter(
            py_fit, r_fit, test_genes, param="theta", rtol=0.1,
        )
        print(f"\nMatch rate: {match_rate:.1%}")
        print(f"Theta correlation: {theta_corr:.4f}")
        assert theta_corr > 0.90, f"Theta correlation too low: {theta_corr:.4f}"
