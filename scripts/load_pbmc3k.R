# load_pbmc3k.R
library(sctransform)
library(Matrix)

url <- "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
tmp <- tempfile(fileext = ".tar.gz")
download.file(url, tmp, mode = "wb")
untar(tmp, exdir = tempdir())

data_dir <- file.path(tempdir(), "filtered_gene_bc_matrices", "hg19")
matrix <- as(readMM(file.path(data_dir, "matrix.mtx")), "dgCMatrix")
genes <- read.delim(file.path(data_dir, "genes.tsv"), header = FALSE, stringsAsFactors = FALSE)
barcodes <- read.delim(file.path(data_dir, "barcodes.tsv"), header = FALSE, stringsAsFactors = FALSE)
rownames(matrix) <- genes$V2
colnames(matrix) <- barcodes$V1

matrix <- matrix[rowSums(matrix > 0) >= 3, colSums(matrix > 0) >= 200]
cat(sprintf("Matrix: %d genes x %d cells\n", nrow(matrix), ncol(matrix)))

set.seed(42)