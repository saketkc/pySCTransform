"""Main module."""
import concurrent.futures
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
import tensorflow as tf
import tensorflow_probability as tfp
from patsy import dmatrix
from statsmodels.api import GLM
from statsmodels.nonparametric.kernel_regression import KernelReg
from tqdm import tqdm

logging.captureWarnings(True)

from statsmodels.tools.sm_exceptions import ConvergenceWarning


# TODO support this
@tf.function(autograph=False)
def tfp_fit(response, model_matrix):
    return tfp.glm.fit(
        model_matrix=model_matrix,
        response=response,
        model=tfp.glm.NegativeBinomial(),
        maximum_iterations=100,
    )


def do_tfp_fit(response, model_matrix, design_info_cols):
    [model_coefficients, linear_response, is_converged, num_iter] = [
        t.numpy()
        for t in tfp_fit(
            response,
            model_matrix,
        )
    ]
    # response.reshape(
    # response.shape[0],
    #        ),
    theta = linear_response.mean() ** 2 / (
        linear_response.var() - linear_response.mean()
    )
    if theta < 0:
        theta = -theta
    theta = 1 / theta
    params = dict(zip(design_info_cols, model_coefficients))
    params["theta"] = theta
    return params


import warnings

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)


def row_gmean(umi, gmean_eps):
    gmean = np.exp(np.log(umi + gmean_eps).mean(1)) - gmean_eps
    return gmean


def get_model_params_pergene(gene_umi, model_matrix):  # latent_var, cell_attr):
    params = (
        dm.NegativeBinomial(gene_umi, model_matrix)
        .fit(maxiter=500, tol=1e-3, disp=0)
        .params
    )
    theta = 1 / params[-1]
    params = dict(zip(model_matrix.design_info.column_names, params[:-1]))
    if theta < 0:
        theta = 1e-3
    params["theta"] = theta
    return params


def get_model_params_allgene(umi, model_matrix, threads=12, use_tf=False):

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        if use_tf:
            feed_list = [
                (
                    #tf.convert_to_tensor(row.values),
                    #tf.convert_to_tensor(model_matrix),
                    row.values,
                    model_matrix,
                    model_matrix.design_info.column_names,
                )
                for index, row in umi.iterrows()
            ]
            results = list(
                tqdm(
                    executor.map(lambda p: do_tfp_fit(*p), feed_list),
                    total=len(feed_list),
                )
            )

        else:
            feed_list = [
                (row.values.reshape((-1, 1)), model_matrix)
                for index, row in umi.iterrows()
            ]
            results = list(
                tqdm(
                    executor.map(lambda p: get_model_params_pergene(*p), feed_list),
                    total=len(feed_list),
                )
            )
    params_df = pd.DataFrame(results, index=umi.index)
    params_df.index.name = "gene"

    return params_df


def get_regularized_params(
    model_parameters,
    genes_log10_gmean_step1,
    genes_log10_gmean,
    cell_attr,
    umi,
):
    model_parameters = model_parameters.copy()

    model_parameters_fit = pd.DataFrame(
        np.nan, index=genes_log10_gmean.index, columns=model_parameters.columns
    )
    exog_predict = genes_log10_gmean.values
    for column in model_parameters.columns:
        endog = model_parameters.loc[genes_log10_gmean_step1.index, column].values
        exog_fit = genes_log10_gmean_step1.values

        reg = KernelReg(endog=endog, exog=exog_fit, var_type="c")
        model_parameters_fit[column] = reg.fit(exog_predict)[0]
    return model_parameters_fit


def get_residuals(umi, model_matrix, model_parameters_fit):

    subset = model_parameters_fit[model_matrix.design_info.column_names]
    theta = model_parameters_fit["theta"]

    mu = np.exp(subset.dot(model_matrix.T))
    mu.columns = umi.columns

    variance = mu + (mu ** 2).divide(theta, axis=0)

    return (umi - mu) / np.sqrt(variance)


def vst(
    umi,
    n_cells=5000,
    latent_var=["log10_umi"],
    gmean_eps=1e-6,
    min_cells=5,
    n_genes=2000,
    threads=12,
    use_tf=False,
):
    """

    Parameters
    ----------
    umi: pd.DataFrame
         pandas DataFrame (dense)
         with genes as rows and cells as columns
         (same as Seurat)

    """
    n_cells = min(n_cells, umi.shape[1])

    log10_umi = np.log10(umi.sum(0))
    expressed_genes = (umi > 0).sum(0)
    log10_expressed_genes = np.log10(expressed_genes)

    cell_attr = pd.DataFrame(index=umi.columns)
    cell_attr["log10_umi"] = log10_umi
    # this is referrred to as gene in SCTransform
    cell_attr["log10_gene"] = log10_expressed_genes
    cell_attr["umi_per_gene"] = log10_umi / expressed_genes
    cell_attr["log10_umi_per_gene"] = np.log10(cell_attr["umi_per_gene"])

    cells_step1 = np.random.choice(umi.columns.tolist(), size=n_cells, replace=False)
    genes_cell_count = (umi.loc[:, cells_step1] > 0).sum(1)
    genes = genes_cell_count >= min_cells
    umi = umi.loc[genes]

    genes_log10_gmean = np.log10(row_gmean(umi, gmean_eps))

    # Step 1: Estimate theta

    data_step1 = cell_attr  # .loc[cells_step1]
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    model_parameters = get_model_params_allgene(
        umi, model_matrix, threads, use_tf
    )  # latent_var, cell_attr)
    ## return model_params
    # Step 2: Do regularization

    # Remove high disp genes
    # Not optimal
    # TODO: Fix
    genes_log10_gmean_step1 = genes_log10_gmean[model_parameters.theta < 10]
    model_parameters_fit = get_regularized_params(
        model_parameters,
        genes_log10_gmean_step1,
        genes_log10_gmean,
        cell_attr,
        umi,
    )

    # Step 3: Calculate residuals
    residuals = get_residuals(umi, model_matrix, model_parameters_fit)

    return {
        "residuals": residuals,
        "model_parameters": model_parameters,
        "model_parameters_fit": model_parameters_fit,
        "genes_log10_gmean_step1": genes_log10_gmean_step1,
        "genes_log10_gmean": genes_log10_gmean,
        "cell_attr": cell_attr,
    }
