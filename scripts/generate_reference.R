#!/usr/bin/env Rscript
# Generate sctransform reference output with NO gene subsampling

# Download PBMC3K
source("scripts/load_pbmc3k.R")

# Run sctransform with NO gene subsampling
vst_out <- vst(
  matrix,
  n_genes = NULL,
  method = "poisson",
  theta_estimation_fun = "theta.ml",
  verbosity = 2)
residuals_sample <- vst_out$y[1:500,]

# Save outputs
write.csv(vst_out$model_pars, "data/r_model_pars.csv")
write.csv(vst_out$model_pars_fit, "data/r_model_pars_fit.csv")
write.csv(as.matrix(residuals_sample), "./data/r_residuals.csv")