library(sctransform)
library(Matrix)
library(Seurat)

options(future.globals.maxSize = 2000 * 1024^2)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript generate_integrated_reference.R <slice1_dir> <slice2_dir> <out_dir>")
}

slice1_dir <- args[1]
slice2_dir <- args[2]
out_dir <- args[3]

slice1 <- Load10X_Spatial(slice1_dir)
slice1$slice <- "slice1"
slice2 <- Load10X_Spatial(slice2_dir)
slice2$slice <- "slice2"

combined <- merge(slice1, slice2)
combined <- JoinLayers(combined)
counts <- LayerData(combined, layer = "counts")

cell_attr <- data.frame(
  row.names = colnames(counts),
  slice = combined$slice
)

vst_out <- sctransform::vst(
  umi = counts,
  cell_attr = cell_attr,
  latent_var = c("log_umi"),
  batch_var = "slice",
  method = "poisson",
  theta_estimation_fun = "theta.ml",
  min_cells = 10,
  n_genes = 3000,
  n_cells = NULL,
  verbosity = 2
)

# HVG ranking by residual variance
gene_residual_vars <- apply(vst_out$y, 1, var)
top_genes <- names(sort(gene_residual_vars, decreasing = TRUE))[1:min(10000, length(gene_residual_vars))]
residuals <- vst_out$y[top_genes, ]

# Output directory
dir.create(out_dir, showWarnings = FALSE)
cat("Saving outputs to:", out_dir, "\n")

write.csv(vst_out$model_pars_fit, file.path(out_dir, "model_pars_fit.csv"))
write.csv(vst_out$model_pars, file.path(out_dir, "model_pars.csv"))
residuals_subset <- residuals[1:min(500, nrow(residuals)), ]
write.csv(as.matrix(residuals_subset), file.path(out_dir, "model_residuals.csv"))

hvg_df <- data.frame(
  gene = names(sort(gene_residual_vars, decreasing = TRUE)),
  residual_variance = sort(gene_residual_vars, decreasing = TRUE)
)
write.csv(hvg_df, file.path(out_dir, "model_hvg_ranking.csv"), row.names = FALSE)
write.csv(cell_attr, file.path(out_dir, "model_cell_attr.csv"))
