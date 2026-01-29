"""Shared utilities for downloading and caching test data."""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


def download_pbmc3k(cache_dir: Path) -> tuple:
    """Download PBMC3k data from 10x Genomics."""
    url = ("https://cf.10xgenomics.com/samples/cell/pbmc3k" +
           "/pbmc3k_filtered_gene_bc_matrices.tar.gz")
    tar_path = cache_dir / "data.tar.gz"
    data_dir = cache_dir / "filtered_gene_bc_matrices" / "hg19"

    # Download if not already cached
    if not data_dir.exists():
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent':
                    'Mozilla/5.0 (Windows NT 4.51; Win64; x64) AppleWebKit/537.36'
            },
        )
        with urllib.request.urlopen(request) as response:
            with open(tar_path, 'wb') as f:
                f.write(response.read())

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)

    # Load data
    matrix = scipy.io.mmread(data_dir / "matrix.mtx").tocsc()
    genes = pd.read_csv(data_dir / "genes.tsv", sep="\t", header=None)[1].tolist()
    cells = pd.read_csv(data_dir / "barcodes.tsv", sep="\t", header=None)[0].tolist()

    return matrix, genes, cells


def filter_pbmc3k(
        matrix, genes: list, cells: list, min_genes: int = 3, min_cells: int = 200,
) -> tuple:
    """Filter genes and cells by minimum counts."""
    gene_mask = np.array((matrix > 0).sum(axis=1)).flatten() >= min_genes
    cell_mask = np.array((matrix > 0).sum(axis=0)).flatten() >= min_cells

    matrix = matrix[gene_mask][:, cell_mask]
    genes = [g for g, m in zip(genes, gene_mask) if m]
    cells = [c for c, m in zip(cells, cell_mask) if m]

    return matrix, genes, cells


def get_pbmc3k_filtered(
        cache_dir: Path, min_genes: int = 3, min_cells: int = 200,
) -> tuple:
    """Download and filter PBMC3k data."""
    matrix, genes, cells = download_pbmc3k(cache_dir)
    return filter_pbmc3k(matrix, genes, cells, min_genes, min_cells)


def load_r_reference(filepath: str, deduplicate: bool = True) -> pd.DataFrame:
    """Load R reference data from CSV, optionally removing duplicate indices."""
    df = pd.read_csv(filepath, index_col=0)
    if deduplicate:
        df = df[~df.index.duplicated(keep='first')]
    return df


def compare_params(
        py_value: float,
        r_value: float,
        atol: float = 0.001,
        rtol: float | None = None,
) -> tuple[bool, float]:
    """
    Compare a Python parameter value against an R reference value.

    Parameters
    ----------
    py_value : float
        Python computed value.
    r_value : float
        R reference value.
    atol : float
        Absolute tolerance (used when rtol is None).
    rtol : float, optional
        Relative tolerance. If provided, uses relative comparison.

    Returns
    -------
    tuple[bool, float]
        (is_match, difference) where difference is absolute or relative.
    """
    # Handle infinities
    if np.isinf(r_value) and np.isinf(py_value):
        return True, 0.0
    if np.isinf(r_value) or np.isinf(py_value):
        return False, np.inf

    if rtol is not None:
        diff = abs(py_value - r_value) / max(abs(r_value), 1e-10)
        return diff < rtol, diff
    else:
        diff = abs(py_value - r_value)
        return diff < atol, diff


def compute_correlation(
        x: np.ndarray, y: np.ndarray, finite_only: bool = True,
) -> float:
    """
    Compute Pearson correlation between two arrays.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays to correlate.
    finite_only : bool
        If True, only use finite values from both arrays.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if finite_only:
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

    if len(x) < 2:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def print_comparison_header(columns: list[tuple[str, int]], total_width: int = 85):
    """Print a formatted comparison table header."""
    header = "".join(f"{name:>{width}}" for name, width in columns)
    print("=" * total_width)
    print(header)
    print("-" * total_width)


def print_comparison_footer(total_width: int = 85):
    """Print a formatted comparison table footer."""
    print("=" * total_width)
