import numpy as np
import pandas as pd
from scipy import sparse

from pysctransform.pysctransform import get_regularized_params, row_gmean


class TestBatchCamping:
    def _make_batch_inputs(
        self,
        n_step1_genes=30,
        n_total_genes=50,
        n_cells=60,
        zero_genes_in_batch_b=None,
        high_expr_genes_in_batch_b=None,
    ):
        """Create inputs for get_regularized_params with batch_var.

        Parameters
        ----------
        zero_genes_in_batch_b : list of int, optional
            Gene indices (within step1) to zero out in batch B,
            forcing -inf in log10(gmean).
        high_expr_genes_in_batch_b : range or list of int, optional
            Gene indices (non-step1) with very high expression in batch B,
            forcing extrapolation beyond step1 gmean range.
        """
        rng = np.random.RandomState(42)
        n_per_batch = n_cells // 2
        batch_assignments = np.array(["A"] * n_per_batch + ["B"] * n_per_batch)
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        gene_names = np.array([f"gene_{i}" for i in range(n_total_genes)])

        cell_attr = pd.DataFrame({"batch": batch_assignments}, index=cell_names)
        cell_attr["log10_umi"] = rng.uniform(3.0, 4.0, n_cells)

        umi = np.zeros((n_total_genes, n_cells))
        for i in range(n_step1_genes):
            umi[i, :] = rng.poisson(lam=50, size=n_cells)

        # Non-step1 genes: moderate by default
        for i in range(n_step1_genes, n_total_genes):
            umi[i, :] = rng.poisson(lam=50, size=n_cells)

        # Zero out specific step1 genes in batch B to force -inf
        if zero_genes_in_batch_b is not None:
            for i in zero_genes_in_batch_b:
                umi[i, n_per_batch:] = 0

        # Make specific non-step1 genes very high in batch B
        if high_expr_genes_in_batch_b is not None:
            for i in high_expr_genes_in_batch_b:
                umi[i, n_per_batch:] = rng.poisson(lam=50000, size=n_per_batch)

        umi = sparse.csr_matrix(umi)
        genes_step1 = gene_names[:n_step1_genes]

        genes_log10_gmean = np.log10(row_gmean(umi, gmean_eps=1))
        step1_idx = np.arange(n_step1_genes)
        genes_log10_gmean_step1 = genes_log10_gmean[step1_idx]

        # Batch-specific columns to trigger the batch code path
        model_params = pd.DataFrame(index=genes_step1)
        model_params["C(batch)[A]"] = rng.normal(-5, 0.5, n_step1_genes)
        model_params["C(batch)[B]"] = rng.normal(-5, 0.5, n_step1_genes)
        model_params["C(batch)[A]:log10_umi"] = rng.normal(
            np.log(10), 0.05, n_step1_genes,
        )
        model_params["C(batch)[B]:log10_umi"] = rng.normal(
            np.log(10), 0.05, n_step1_genes,
        )
        model_params["theta"] = rng.uniform(10, 100, n_step1_genes)
        model_params["od_factor"] = np.log10(
            1 + np.power(10, genes_log10_gmean_step1) / model_params["theta"].values,
        )

        return dict(
            model_parameters=model_params,
            genes=gene_names,
            genes_step1=genes_step1,
            genes_log10_gmean_step1=genes_log10_gmean_step1,
            genes_log10_gmean=genes_log10_gmean,
            cell_attr=cell_attr,
            umi=umi,
        )

    def test_inf_replacement_affects_regularization(self):
        inputs = self._make_batch_inputs(
            zero_genes_in_batch_b=[0, 1, 2],
        )

        result = get_regularized_params(
            **inputs,
            batch_var="batch",
            theta_regularization="od_factor",
        )

        for col in result.columns:
            if col == "theta":
                continue
            vals = result[col].values
            assert np.all(np.isfinite(vals)), (
                f"Column '{col}' has non-finite values: "
                f"{vals[~np.isfinite(vals)]}"
            )

    def test_upper_clamp_prevents_nan_from_extrapolation(self):
        inputs = self._make_batch_inputs(
            high_expr_genes_in_batch_b=range(30, 50),
        )

        result = get_regularized_params(
            **inputs,
            batch_var="batch",
            theta_regularization="od_factor",
        )

        # Check batch-B specific columns for NaN
        batch_b_cols = [
            c for c in result.columns
            if "B" in c and c not in ("theta", "od_factor")
        ]
        assert len(batch_b_cols) > 0, "No batch-B columns found"

        extra_genes = inputs["genes"][30:]
        for col in batch_b_cols:
            extra_vals = result.loc[extra_genes, col].values
            n_nan = np.sum(np.isnan(extra_vals))
            assert n_nan == 0, (
                f"Column '{col}' has {n_nan}/{len(extra_vals)} NaN values "
                f"in high-expression genes (extrapolation not clamped?)"
            )

    def test_both_guards_with_extreme_batch_imbalance(self):
        inputs = self._make_batch_inputs(
            zero_genes_in_batch_b=[0, 1],
            high_expr_genes_in_batch_b=range(30, 50),
        )

        result = get_regularized_params(
            **inputs,
            batch_var="batch",
            theta_regularization="od_factor",
        )

        for col in result.columns:
            if col == "theta":
                continue
            vals = result[col].values
            n_nonfinite = np.sum(~np.isfinite(vals))
            assert n_nonfinite == 0, (
                f"Column '{col}' has {n_nonfinite} non-finite values"
            )

        assert np.all(result["theta"].values > 0)
