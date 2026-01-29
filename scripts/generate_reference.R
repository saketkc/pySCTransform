#!/usr/bin/env Rscript
# Generate sctransform reference output with NO gene subsampling

library(sctransform)
library(Matrix)

# Download PBMC3k data
url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
tmp <- tempfile(fileext = ".tar.gz")
download.file(url, tmp, mode = "wb")
untar(tmp, exdir = tempdir())

# Read 10x data
data_dir <- file.path(tempdir(), "filtered_gene_bc_matrices", "hg19")
matrix <- as(readMM(file.path(data_dir, "matrix.mtx")), "dgCMatrix")
genes <- read.delim(file.path(data_dir, "genes.tsv"), header = FALSE, stringsAsFactors = FALSE)
barcodes <- read.delim(file.path(data_dir, "barcodes.tsv"), header = FALSE, stringsAsFactors = FALSE)
rownames(matrix) <- genes$V2
colnames(matrix) <- barcodes$V1

# Filter
matrix <- matrix[rowSums(matrix > 0) >= 3, colSums(matrix > 0) >= 200]
cat(sprintf("Matrix: %d genes x %d cells\n", nrow(matrix), ncol(matrix)))

# Run sctransform with NO gene subsampling
set.seed(42)
vst_out <- vst(matrix, n_genes = NULL, method = "poisson", verbosity = 2)

# Save raw parameters (ALL genes)
write.csv(vst_out$model_pars, "data/r_model_pars_all.csv")
cat(sprintf("Saved %d genes to r_model_pars_all.csv\n", nrow(vst_out$model_pars)))