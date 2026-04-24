"""
Compare pysctransform output against R sctransform reference.

Usage:
    python compare_integrated_reference.py <slice1_dir> <slice2_dir> <r_reference_dir>
"""
import sys
import os

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import hstack

import pysctransform


def load_10x_matrix(path):
    mat_dir = os.path.join(path, "filtered_feature_bc_matrix")
    matrix = mmread(os.path.join(mat_dir, "matrix.mtx.gz")).T.tocsc()
    barcodes = pd.read_csv(
        os.path.join(mat_dir, "barcodes.tsv.gz"), header=None, sep="\t",
    )[0].values
    features = pd.read_csv(
        os.path.join(mat_dir, "features.tsv.gz"), header=None, sep="\t",
    )
    gene_names = features[1].values
    return matrix, gene_names, barcodes


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_integrated_reference.py <slice1_dir> <slice2_dir> <r_reference_dir>")
        sys.exit(1)

    slice1_dir = sys.argv[1]
    slice2_dir = sys.argv[2]
    ref_dir = sys.argv[3]

    # Load data
    print("Loading slice 1...")
    mat1, genes1, barcodes1 = load_10x_matrix(slice1_dir)
    print("Loading slice 2...")
    mat2, genes2, barcodes2 = load_10x_matrix(slice2_dir)

    # Use R's cell names to match Seurat's naming
    r_cell_attr = pd.read_csv(os.path.join(ref_dir, "model_cell_attr.csv"), index_col=0)
    r_slice1_cells = r_cell_attr.index[r_cell_attr["slice"] == "slice1"].tolist()
    r_slice2_cells = r_cell_attr.index[r_cell_attr["slice"] == "slice2"].tolist()

    assert len(r_slice1_cells) == len(barcodes1), \
        f"Cell count mismatch slice1: R={len(r_slice1_cells)} vs data={len(barcodes1)}"
    assert len(r_slice2_cells) == len(barcodes2), \
        f"Cell count mismatch slice2: R={len(r_slice2_cells)} vs data={len(barcodes2)}"

    cell_names = np.array(r_slice1_cells + r_slice2_cells)

    # Combine on common genes
    common_genes = np.intersect1d(genes1, genes2)
    print(f"Common genes: {len(common_genes)}")

    idx1 = np.array([np.where(genes1 == g)[0][0] for g in common_genes])
    idx2 = np.array([np.where(genes2 == g)[0][0] for g in common_genes])

    combined = hstack([mat1[:, idx1].T, mat2[:, idx2].T], format="csc")
    gene_names = common_genes

    cell_attr = pd.DataFrame(
        {"slice": ["slice1"] * len(barcodes1) + ["slice2"] * len(barcodes2)},
        index=cell_names,
    )

    print(f"Combined: {combined.shape[0]} genes x {combined.shape[1]} cells")

    # Run pysctransform
    print("Running pysctransform...")
    vst_out = pysctransform.vst(
        combined,
        gene_names=gene_names.tolist(),
        cell_names=cell_names.tolist(),
        cell_attr_extra=cell_attr[["slice"]],
        method="theta_ml",
        n_cells=None,
        n_genes=3000,
        batch_var="slice",
        min_cells=10,
        verbosity=True,
    )

    py_fit = vst_out["model_parameters_fit"]
    print(f"\nod_factor stats:")
    od = py_fit["od_factor"]
    print(f"  min: {od.min():.6f}, max: {od.max():.6f}, mean: {od.mean():.6f}")
    print(f"  NaN count: {od.isna().sum()}")
    print(f"  <= 0 count: {(od <= 0).sum()}")
    print(f"  > 0 count: {(od > 0).sum()}")

    s1_col = "C(slice)[slice1]"
    if s1_col in py_fit.columns:
        s1 = py_fit[s1_col]
        print(f"\nslice1 intercept stats:")
        print(f"  min: {s1.min():.6f}, max: {s1.max():.6f}")
        print(f"  NaN count: {s1.isna().sum()}, finite: {np.isfinite(s1).sum()}")

    residuals = pysctransform.get_hvg_residuals(vst_out, var_features_n=10000)

    # ---- Compare fitted parameters ----
    print("\n=== Fitted Parameters ===")
    r_fit = pd.read_csv(os.path.join(ref_dir, "model_pars_fit.csv"), index_col=0)
    py_fit = vst_out["model_parameters_fit"]

    common_fit_genes = sorted(set(r_fit.index) & set(py_fit.index))
    print(f"Common genes in fitted params: {len(common_fit_genes)}")

    # After running vst
    print(f"\nPython fitted param columns: {list(py_fit.columns)}")
    print(f"R fitted param columns: {list(r_fit.columns)}")

    # Also check model matrix columns
    print(
        f"Python model matrix columns: {list(vst_out['model_matrix'].design_info.column_names)}")

    if len(common_fit_genes) > 10:
        # Build column name mapping: R name -> Python name
        col_map = {}
        for r_col in r_fit.columns:
            if r_col == "theta":
                col_map[r_col] = "theta"
                continue
            # R: "sliceslice1" -> Python: "C(slice)[slice1]"
            # R: "log_umi:sliceslice1" -> Python: "log10_umi:C(slice)[slice1]"
            py_col = r_col
            # Handle interaction terms first
            if ":" in r_col:
                parts = r_col.split(":")
                latent_part = parts[0]  # e.g. "log_umi"
                batch_part = parts[1]  # e.g. "sliceslice1"
            else:
                latent_part = None
                batch_part = r_col

            # Map batch part: "sliceslice1" -> "C(slice)[slice1]"
            # R concatenates batch_var name + level, e.g. "slice" + "slice1" = "sliceslice1"
            batch_var_name = "slice"
            if batch_part.startswith(batch_var_name):
                level = batch_part[len(batch_var_name):]
                py_batch = f"C(slice)[{level}]"
            else:
                py_batch = batch_part

            if latent_part is not None:
                # R uses log_umi, Python uses log10_umi
                py_latent = latent_part.replace("log_umi", "log10_umi")
                py_col = f"{py_latent}:{py_batch}"
            else:
                py_col = py_batch

            col_map[r_col] = py_col

        print(f"Column mapping: {col_map}")

        # Theta correlation
        r_theta = r_fit.loc[common_fit_genes, "theta"].values.astype(float)
        py_theta = py_fit.loc[common_fit_genes, "theta"].values.astype(float)
        finite = np.isfinite(r_theta) & np.isfinite(py_theta)
        if finite.sum() > 10:
            theta_corr = np.corrcoef(r_theta[finite], py_theta[finite])[0, 1]
            theta_rel_diff = np.mean(
                np.abs(r_theta[finite] - py_theta[finite])
                / np.maximum(r_theta[finite], 0.001)
            )
            print(f"  Theta correlation: {theta_corr:.4f}")
            print(f"  Theta mean relative diff: {theta_rel_diff:.4f}")
            print(f"  Finite thetas: {finite.sum()}/{len(finite)}")
        else:
            print(f"  Not enough finite theta values to compare")
            print(
                f"  R finite: {np.isfinite(r_theta).sum()}, Py finite: {np.isfinite(py_theta).sum()}")
            print(
                f"  R theta range: [{np.nanmin(r_theta):.4f}, {np.nanmax(r_theta):.4f}]")
            print(
                f"  Py theta range: [{np.nanmin(py_theta):.4f}, {np.nanmax(py_theta):.4f}]")

        # Compare each coefficient column using mapping
        for r_col, py_col in col_map.items():
            if r_col == "theta":
                continue
            if py_col not in py_fit.columns:
                print(
                    f"  Column '{r_col}' -> '{py_col}' not found in Python output, skipping")
                continue

            r_vals = r_fit.loc[common_fit_genes, r_col].values.astype(float)
            py_vals = py_fit.loc[common_fit_genes, py_col].values.astype(float)
            finite = np.isfinite(r_vals) & np.isfinite(py_vals)
            if finite.sum() > 10:
                corr = np.corrcoef(r_vals[finite], py_vals[finite])[0, 1]
                print(f"  {r_col} -> {py_col} correlation: {corr:.4f}")

    # ---- Compare residuals ----
    print("\n=== Residuals ===")
    r_resid = pd.read_csv(os.path.join(ref_dir, "model_residuals.csv"), index_col=0)
    py_full_resid = vst_out["residuals"]

    common_resid_genes = sorted(set(r_resid.index) & set(py_full_resid.index))
    common_resid_cells = sorted(set(r_resid.columns) & set(py_full_resid.columns))
    print(f"Common genes in residuals: {len(common_resid_genes)}")
    print(f"Common cells in residuals: {len(common_resid_cells)}")

    if len(common_resid_genes) > 0 and len(common_resid_cells) > 0:
        test_genes = common_resid_genes[:100]
        test_cells = common_resid_cells[:200]

        correlations = []
        low_corr = []
        for gene in test_genes:
            r_vals = r_resid.loc[gene, test_cells].values.astype(float)
            py_vals = py_full_resid.loc[gene, test_cells].values.astype(float)
            if np.std(r_vals) > 0 and np.std(py_vals) > 0:
                corr = np.corrcoef(r_vals, py_vals)[0, 1]
                correlations.append(corr)
                if corr < 0.9:
                    low_corr.append((gene, corr))

        if correlations:
            print(f"  Per-gene correlation (n={len(correlations)}):")
            print(f"    Mean: {np.mean(correlations):.4f}")
            print(f"    Median: {np.median(correlations):.4f}")
            print(f"    Min: {np.min(correlations):.4f}")
            if low_corr:
                print(f"    Genes with corr < 0.9: {len(low_corr)}")
                for g, c in low_corr[:5]:
                    print(f"      {g}: {c:.4f}")

        # Overall correlation
        r_flat = r_resid.loc[test_genes, test_cells].values.astype(float).flatten()
        py_flat = py_full_resid.loc[test_genes, test_cells].values.astype(float).flatten()
        overall_corr = np.corrcoef(r_flat, py_flat)[0, 1]
        print(f"  Overall correlation: {overall_corr:.4f}")

    # ---- Compare HVG ranking ----
    print("\n=== HVG Ranking ===")
    r_hvg = pd.read_csv(os.path.join(ref_dir, "model_hvg_ranking.csv"))
    r_hvg_genes = r_hvg["gene"].tolist()

    py_gene_attr = vst_out["gene_attr"].sort_values("residual_variance", ascending=False)
    py_hvg_genes = py_gene_attr.index.tolist()

    for n in [100, 500, 1000, 3000, 5000]:
        if n > len(r_hvg_genes) or n > len(py_hvg_genes):
            continue
        r_set = set(r_hvg_genes[:n])
        py_set = set(py_hvg_genes[:n])
        overlap = len(r_set & py_set)
        jaccard = overlap / len(r_set | py_set)
        print(f"  Top {n}: overlap={overlap}/{n} ({overlap/n*100:.1f}%), Jaccard={jaccard:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()