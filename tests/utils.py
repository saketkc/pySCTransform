"""Shared utilities for downloading and caching test data."""
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
import scipy.io


def download_pbmc3k(cache_dir: Path) -> tuple:
    url = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    tar_path = cache_dir / "data.tar.gz"
    data_dir = cache_dir / "filtered_gene_bc_matrices" / "hg19"

    # Download if not already cached
    if not data_dir.exists():
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 4.51; Win64; x64) AppleWebKit/537.36'
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
        matrix, genes: list, cells: list, min_genes: int = 3, min_cells: int = 200
        ) -> tuple:
    gene_mask = np.array((matrix > 0).sum(axis=1)).flatten() >= min_genes
    cell_mask = np.array((matrix > 0).sum(axis=0)).flatten() >= min_cells

    matrix = matrix[gene_mask][:, cell_mask]
    genes = [g for g, m in zip(genes, gene_mask) if m]
    cells = [c for c, m in zip(cells, cell_mask) if m]

    return matrix, genes, cells


def get_pbmc3k_filtered(
        cache_dir: Path, min_genes: int = 3, min_cells: int = 200
        ) -> tuple:
    matrix, genes, cells = download_pbmc3k(cache_dir)
    return filter_pbmc3k(matrix, genes, cells, min_genes, min_cells)