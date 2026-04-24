#!/usr/bin/env Rscript
# Generate sctransform reference output WITH batch variable

# Download PBMC3K
source("scripts/load_pbmc3k.R")

# Create a synthetic batch variable by splitting cells in half
n_cells <- ncol(matrix)
batch_labels <- rep("A", n_cells)
batch_labels[sample(n_cells, floor(n_cells / 2))] <- "B"
cell_attr <- data.frame(batch = factor(batch_labels), row.names = colnames(matrix))

cat(sprintf("Batch A: %d cells, Batch B: %d cells\n",
            sum(batch_labels == "A"), sum(batch_labels == "B")))

# Run sctransform with batch variable, no gene subsampling
vst_out <- vst(
  matrix,
  cell_attr = cell_attr,
  batch_var = "batch",
  n_genes = NULL,
  n_cells = NULL,
  min_cells = 10,
  method = "poisson",
  theta_estimation_fun = "theta.ml",
  verbosity = 2
)

residuals_sample <- vst_out$y[1:500,]

# Save outputs
write.csv(vst_out$model_pars, "data/r_batch_model_pars.csv")
write.csv(vst_out$model_pars_fit, "data/r_batch_model_pars_fit.csv")
write.csv(as.matrix(residuals_sample), "data/r_batch_residuals.csv")
write.csv(cell_attr, "data/r_batch_cell_attr.csv")