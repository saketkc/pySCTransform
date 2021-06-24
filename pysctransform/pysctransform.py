"""Main module."""
import time
import warnings

from KDEpy import FFTKDE
from scipy import interpolate
from scipy import sparse
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import concurrent.futures
import logging

import numpy as npy
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from joblib import Parallel
from joblib import delayed
from patsy import dmatrix
from scipy import stats
from scipy.sparse import csr_matrix
from statsmodels.api import GLM
from statsmodels.nonparametric.kernel_regression import KernelReg
from tqdm import tqdm

logging.captureWarnings(True)


from .fit import alpha_lbfgs
from .fit import estimate_mu_glm
from .fit import estimate_mu_poisson
from .fit import theta_lbfgs
from .fit import theta_ml

# import jax
# import jax.numpy as jnpy
# from .fit import jax_alpha_lbfgs
# from .jax_theta_ml import jax_theta_ml
# from .jax_lbfgs import fit_nbinom_lbfgs_autograd
# from .jax_bfgs import fit_nbinom_bfgs_alpha_jit
# from .jax_bfgs import fit_nbinom_bfgs_jit
from .r_bw import bw_SJr
from .r_bw import is_outlier_r
from .r_bw import ksmooth


def is_outlier_naive(x, snr_threshold=25):
    """
    Mark points as outliers
    """
    if len(x.shape) == 1:
        x = x[:, None]
    median = npy.median(x, axis=0)
    diff = npy.sum((x - median) ** 2, axis=-1)
    diff = npy.sqrt(diff)
    med_abs_deviation = npy.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > snr_threshold


def sparse_var(X, axis=None):
    X2 = X.copy()
    X2.data **= 2
    return X2.mean(axis) - npy.square(X2.mean(axis))


def bwSJ(genes_log10_gmean_step1, bw_adjust=3):
    # See https://kdepy.readthedocs.io/en/latest/bandwidth.html
    fit = FFTKDE(kernel="gaussian", bw="ISJ").fit(npy.asarray(genes_log10_gmean_step1))
    _ = fit.evaluate()
    bw = fit.bw * bw_adjust
    return npy.array([bw], dtype=float)


def robust_scale(x):
    return (x - npy.median(x)) / (
        stats.median_absolute_deviation(x) + npy.finfo(float).eps
    )


def robust_scale_binned(y, x, breaks):
    bins = pd.cut(x=x, bins=breaks, ordered=True)

    # categories = bins.categories
    # bins = npy.digitize(x=x, bins=breaks)
    df = pd.DataFrame({"x": y, "bins": bins})
    tmp = df.groupby(["bins"]).apply(robust_scale)
    order = df["bins"].argsort()
    tmp = tmp.loc[order]  # sort_values(by=["bins"])
    score = tmp["x"]
    return score


def is_outlier(y, x, th=10):
    bin_width = (npy.nanmax(x) - npy.nanmin(x)) * bwSJ(x, bw_adjust=1 / 2)
    eps = npy.finfo(float).eps * 10
    bin_width = bin_width[0]
    breaks1 = npy.arange(
        start=npy.nanmin(x) - eps, stop=npy.nanmax(x) + bin_width, step=bin_width
    )
    breaks2 = npy.arange(
        start=npy.nanmin(x) - eps - bin_width / 2,
        stop=npy.nanmax(x) + bin_width,
        step=bin_width,
    )
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return npy.vstack((npy.abs(score1), npy.abs(score2))).min(0) > th


def make_cell_attr(umi, cell_names):
    assert umi.shape[1] == len(cell_names)
    total_umi = npy.squeeze(npy.asarray(umi.sum(0)))
    log10_umi = npy.log10(total_umi)
    expressed_genes = npy.squeeze(npy.asarray((umi > 0).sum(0)))
    log10_expressed_genes = npy.log10(expressed_genes)
    cell_attr = pd.DataFrame({"umi": total_umi, "log10_umi": log10_umi})
    cell_attr.index = cell_names
    cell_attr["n_expressed_genes"] = expressed_genes
    # this is referrred to as gene in SCTransform
    cell_attr["log10_gene"] = log10_expressed_genes
    cell_attr["umi_per_gene"] = log10_umi / expressed_genes
    cell_attr["log10_umi_per_gene"] = npy.log10(cell_attr["umi_per_gene"])
    return cell_attr


def row_gmean(umi, gmean_eps=1):
    gmean = npy.exp(npy.log(umi + gmean_eps).mean(1)) - gmean_eps
    return gmean


def row_gmean_sparse(umi, gmean_eps=1):

    gmean = npy.asarray(npy.array([row_gmean(x.todense(), gmean_eps)[0] for x in umi]))
    gmean = npy.squeeze(gmean)
    return gmean


def _process_y(y):
    if not isinstance(y, npy.ndarray):
        y = npy.array(y)
    y = npy.asarray(y, dtype=int)
    y = npy.squeeze(y)
    return y


def get_model_params_pergene(
    gene_umi, model_matrix, method="theta_ml"
):
    gene_umi = _process_y(gene_umi)
    if method == "sm_nb":
        model = dm.NegativeBinomial(gene_umi, model_matrix, loglike_method="nb2")
        params = model.fit(maxiter=50, tol=1e-3, disp=0).params
        theta = 1 / params[-1]
        if theta >= 1e5:
            theta = npy.inf
        params = dict(zip(model_matrix.design_info.column_names, params[:-1]))
        params["theta"] = theta
    elif method == "theta_ml":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = theta_ml(y=gene_umi, mu=mu)
        if theta >= 1e5:
            theta = npy.inf
        params["theta"] = theta
    elif method == "jax_jit":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        gene_umi_jax = jax.device_put(gene_umi)
        mu_jax = jax.device_put(mu)
        theta = float(
            fit_nbinom_bfgs_jit(y=gene_umi_jax, mu=mu_jax).block_until_ready()
        )
        if theta < 0:
            # replace with moment based estimator
            theta = mu ** 2 / (npy.var(gene_umi) - mu)
            if theta < 0:
                theta = npy.inf
        params["theta"] = theta
    elif method == "jax_alpha_jit":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        gene_umi_jax = jax.device_put(gene_umi_jax)
        mu_jax = jax.device_put(mu)
        theta = float(
            fit_nbinom_bfgs_alpha_jit(y=gene_umi_jax, mu=mu_jax).block_until_ready()
        )
        if theta < 0:
            # replace with moment based estimator
            theta = mu ** 2 / (npy.var(gene_umi) - mu)
            if theta < 0:
                theta = npy.inf
        params["theta"] = theta
    elif method == "jax_theta_ml":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = jax_theta_ml(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif method == "alpha_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = alpha_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif method == "jax_alpha_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = jax_alpha_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif method == "theta_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = theta_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif method == "autograd":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = fit_nbinom_lbfgs_autograd(y=gene_umi, mu=mu)
        params["theta"] = theta
    return params


def get_model_params_pergene_glmgp(gene_umi, coldata, design="~ log10_umi"):
    from .fit_glmgp import fit_glmgp
    gene_umi = gene_umi.todense()
    params = fit_glmgp(y=gene_umi, coldata=coldata, design=design)
    return params


def get_model_params_allgene_glmgp(umi, coldata, bin_size=500, threads=12, verbosity=0):

    results = []
    results = Parallel(n_jobs=threads, backend="multiprocessing", batch_size=500)(
        delayed(get_model_params_pergene_glmgp)(row, coldata) for row in umi
    )
    params_df = pd.DataFrame(results)

    return params_df


def get_model_params_allgene(umi, model_matrix, method="fit", threads=12, verbosity=0):

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # TODO this should remain sparse
        # feed_list = [
        #    (row.values.reshape((-1, 1)), model_matrix, method)
        #    for index, row in umi.iterrows()
        # ]
        feed_list = [
            (row.todense().reshape((-1, 1)), model_matrix, method) for row in umi
        ]

        if verbosity:
            results = list(
                tqdm(
                    executor.map(lambda p: get_model_params_pergene(*p), feed_list),
                    total=len(feed_list),
                )
            )
        else:
            results = list(
                executor.map(lambda p: get_model_params_pergene(*p), feed_list)
            )

    params_df = pd.DataFrame(results)

    return params_df


def dds(genes_log10_gmean_step1, grid_points=2 ** 10):
    # density dependent downsampling
    x, y = (
        FFTKDE(kernel="gaussian", bw="silverman")
        .fit(npy.asarray(genes_log10_gmean_step1))
        .evaluate(grid_points=grid_points)
    )
    density = interpolate.interp1d(x=x, y=y, assume_sorted=False)
    sampling_prob = 1 / (density(genes_log10_gmean_step1) + npy.finfo(float).eps)

    return sampling_prob / sampling_prob.sum()


def get_regularized_params(
    model_parameters,
    genes,
    genes_step1,
    genes_log10_gmean_step1,
    genes_log10_gmean,
    cell_attr,
    umi,
    batch_var=None,
    bw_adjust=3,
    gmean_eps=1,
    theta_regularization="od_factor",
    exclude_poisson=False,
    poisson_genes=None,
):
    model_parameters = model_parameters.copy()

    model_parameters_fit = pd.DataFrame(
        npy.nan, index=genes, columns=model_parameters.columns
    )

    """
    exog_predict = genes_log10_gmean#.values
    for column in model_parameters.columns:
        if column == "theta":
            continue
        endog = model_parameters.loc[genes_step1, column].values
        exog_fit = genes_log10_gmean_step1#.values
        bw = bwSJ(genes_log10_gmean_step1, bw_adjust=bw_adjust)#.values)
        reg = KernelReg(endog=endog, exog=exog_fit, var_type="c", reg_type="ll", bw=bw)
        model_parameters_fit[column] = reg.fit(exog_predict)[0]

    """
    x_points_df = pd.DataFrame({"gene_log10_gmean": genes_log10_gmean})
    x_points_df["min_gene_log10_gmean_step1"] = genes_log10_gmean_step1.min()

    x_points_df["x_points"] = npy.nanmax(x_points_df, axis=1)
    x_points_df["max_gene_log10_gmean_step1"] = npy.nanmax(genes_log10_gmean_step1)
    x_points_df["x_points"] = x_points_df[
        ["x_points", "max_gene_log10_gmean_step1"]
    ].min(1)
    x_points = x_points_df["x_points"].values
    for column in model_parameters.columns:
        if column == "theta":
            continue
        endog = model_parameters.loc[genes_step1, column].values
        exog_fit = genes_log10_gmean_step1  # .values
        bw = bwSJ(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
        reg = KernelReg(endog=endog, exog=exog_fit, var_type="c", reg_type="ll", bw=bw)
        fit = reg.fit(x_points)
        model_parameters_fit[column] = npy.squeeze(fit[0])
        # print(bw)
        ##bw = bw_SJr(genes_log10_gmean_step1, bw_adjust=bw_adjust)  # .values)
        ##params = ksmooth(genes_log10_gmean, genes_log10_gmean_step1, endog, bw[0])
        ##index = model_parameters_fit.index.values[params["order"] - 1]
        ##model_parameters_fit.loc[index, column] = params["smoothed"]

    if theta_regularization == "theta":
        theta = npy.power(10, (model_parameters["od_factor"]))
    else:
        theta = npy.power(10, genes_log10_gmean) / (
            npy.power(10, model_parameters_fit["od_factor"]) - 1
        )
    model_parameters_fit["theta"] = theta
    if exclude_poisson:
        # relace theta by inf
        if poisson_genes is not None:
            model_parameters_fit.loc[poisson_genes, "theta"] = npy.inf

    return model_parameters_fit


def pearson_residual(y, mu, theta, min_var=-npy.inf):
    variance = mu + npy.divide(mu ** 2, theta.reshape(-1, 1))
    variance[variance < min_var] = min_var
    pearson_residuals = npy.divide(y - mu, npy.sqrt(variance))
    return pearson_residuals


def deviance_residual(y, mu, theta, weight=1):
    theta = npy.tile(theta.reshape(-1, 1), y.shape[1])
    L = npy.multiply((y + theta), npy.log((y + theta) / (mu + theta)))
    log_mu = npy.log(mu)
    log_y = npy.log(y.maximum(1).todense())
    r = npy.multiply(y.todense(), log_y - log_mu)
    r = 2 * weight * (r - L)
    return npy.multiply(npy.sqrt(r), npy.sign(y - mu))


def get_residuals(
    umi,
    model_matrix,
    model_parameters_fit,
    residual_type="pearson",
    res_clip_range="default",
):

    subset = npy.asarray(
        model_parameters_fit[model_matrix.design_info.column_names].values
    )
    theta = npy.asarray(model_parameters_fit["theta"].values)

    mu = npy.exp(npy.dot(subset, model_matrix.T))
    # variance = mu + npy.divide(mu ** 2, theta.reshape(-1, 1))
    # pearson_residuals = npy.divide(umi - mu, npy.sqrt(variance))
    if residual_type == "pearson":
        residuals = pearson_residual(umi, mu, theta)
    elif residual_type == "deviance":
        residuals = deviance_residual(umi, mu, theta)

    if res_clip_range == "default":
        res_clip_range = npy.sqrt(umi.shape[1] / 30)
        residuals = npy.clip(residuals, a_min=-res_clip_range, a_max=res_clip_range)
    return residuals


def correct(residuals, cell_attr, latent_var, model_parameters_fit, umi):
    # replace value of latent variables with its median
    cell_attr = cell_attr.copy()
    for column in latent_var:
        cell_attr.loc[:, column] = cell_attr.loc[:, column].median()
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    non_theta_columns = [
        x for x in model_matrix.design_info.column_names if x != "theta"
    ]
    coefficients = model_parameters_fit[non_theta_columns]
    theta = model_parameters_fit["theta"].values

    mu = npy.exp(coefficients.dot(model_matrix.T))
    mu = npy.exp(npy.dot(coefficients.values, model_matrix.T))
    variance = mu + npy.divide(mu ** 2, npy.tile(theta.reshape(-1, 1), mu.shape[1]))
    corrected_data = mu + residuals.values * npy.sqrt(variance)
    corrected_data[corrected_data < 0] = 0
    corrected_counts = sparse.csr_matrix(corrected_data.astype(int))

    return corrected_counts


def vst(
    umi,
    gene_names=None,
    cell_names=None,
    n_cells=5000,
    latent_var=["log10_umi"],
    batch_var=None,
    gmean_eps=1,
    min_cells=5,
    n_genes=2000,
    threads=24,
    use_tf=False,
    method="theta_ml",
    theta_given=10,
    theta_regularization="od_factor",
    residual_type="pearson",
    exclude_poisson=False,
    verbosity=0,
):
    """

    Parameters
    ----------
    umi: pd.DataFrame
         pandas DataFrame (dense)
         with genes as rows and cells as columns
         (same as Seurat)

    """
    umi = umi.copy()
    if n_cells is None:
        n_cells = umi.shape[1]
    if n_genes is None:
        n_genes = umi.shape[0]
    n_cells = min(n_cells, umi.shape[1])
    if gene_names is None:
        if not isinstance(umi, pd.DataFrame):
            raise RuntimeError(
                "`gene_names` and `cell_names` are required when umi is not a dataframe"
            )
        else:
            gene_names = umi.index.tolist()
            cell_names = umi.columns.tolist()
            umi = csr_matrix(umi.values)
            # umi.to_numpy()
    if cell_names is None:
        cell_names = [x for x in range(umi.shape[1])]

    gene_names = npy.asarray(gene_names, dtype="U")
    cell_names = npy.asarray(cell_names, dtype="U")
    genes_cell_count = npy.asarray((umi >= 0.01).sum(1))
    min_cells_genes_index = npy.squeeze(genes_cell_count >= min_cells)
    genes = gene_names[min_cells_genes_index]
    cell_attr = make_cell_attr(umi, cell_names)
    if isinstance(umi, pd.DataFrame):
        umi = umi.loc[genes]
    else:
        umi = umi[min_cells_genes_index, :]
    genes_log10_gmean = npy.log10(row_gmean_sparse(umi, gmean_eps=gmean_eps))
    genes_log10_amean = npy.log10(npy.ravel(umi.mean(1)))

    if n_cells is None and n_cells < umi.shape[1]:
        # downsample cells to speed up the first step
        cells_step1_index = npy.random.choice(
            a=npy.arange(len(cell_names), dtype=int), size=n_cells, replace=False
        )
        cells_step1 = cell_names[cells_step1_index]
        genes_cell_count_step1 = (umi[:, cells_step1_index] > 0).sum(1)
        genes_step1 = genes[genes_cell_count_step1 >= min_cells]
        genes_log10_gmean_step1 = npy.log10(
            row_gmean_sparse(umi[genes_step1, cells_step1], gmean_eps=gmean_eps)
        )
        genes_log10_amean_step1 = npy.log10(
            npy.ravel(umi[genes_step1, cells_step1].mean(1))
        )
        umi_step1 = umi[:, cells_step1_index]
    else:
        cells_step1_index = npy.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        genes_log10_amean_step1 = genes_log10_amean
        umi_step1 = umi

    data_step1 = cell_attr.loc[cells_step1]
    if (n_genes is not None) and (n_genes < len(genes_step1)):
        # density-sample genes to speed up the first step
        sampling_prob = dds(genes_log10_gmean_step1)

        genes_step1_index = npy.random.choice(
            a=npy.arange(len(genes_step1)), size=n_genes, replace=False, p=sampling_prob
        )
        genes_step1 = gene_names[genes_step1_index]
        umi_step1 = umi_step1[genes_step1_index, :]  # [:, cells_step1_index]
        genes_log10_gmean_step1 = npy.log10(
            row_gmean_sparse(umi_step1, gmean_eps=gmean_eps)
        )
        genes_log10_amean_step1 = npy.log10(umi_step1.mean(1))

    if method == "offset":
        cells_step1_index = npy.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        genes_log10_amean_step1 = genes_log10_amean
        umi_step1 = umi
    # Step 1: Estimate theta

    if verbosity:
        print("Running Step1")
    start = time.time()
    if batch_var is None:
        model_matrix = dmatrix(" + ".join(latent_var), data_step1)
    else:
        cross_term = "(" + " + ".join(latent_var) + "):" + batch_var
        model_matrix = dmatrix(
            " + ".join(latent_var) + cross_term + " + ".join(batch_var) + " + 0",
            data_step1,
        )

    if method == "offset":
        gene_mean = umi.mean(1)
        mean_cell_sum = npy.mean(umi.sum(0))
        model_parameters = pd.DataFrame(index=genes)
        model_parameters["theta"] = theta_given
        model_parameters["Intercept"] = npy.log(gene_mean) - npy.log(mean_cell_sum)
        model_parameters["log10_umi"] = [npy.log(10)] * len(genes)

    elif method == "glmgp":
        model_parameters = get_model_params_allgene_glmgp(umi_step1, data_step1)
        model_parameters.index = genes_step1
    else:
        model_parameters = get_model_params_allgene(
            umi_step1, model_matrix, method, threads, use_tf
        )  # latent_var, cell_attr)
        model_parameters.index = genes_step1

    gene_attr = pd.DataFrame(index=genes)
    gene_attr["gene_amean"] = umi.mean(1)
    gene_attr["gene_gmean"] = npy.power(10, genes_log10_gmean)
    gene_attr["gene_detectation_rate"] = (
        npy.squeeze(npy.asarray((umi > 0).sum(1))) / umi.shape[1]
    )
    gene_attr["theta"] = model_parameters["theta"]
    gene_attr["gene_variance"] = sparse_var(umi, 1)  # umi.var(1)

    poisson_genes = None
    if exclude_poisson:
        poisson_genes = gene_attr[
            gene_attr["gene_amean"] >= gene_attr["gene_variance"]
        ].index.tolist()
        if verbosity:
            print("Found ", len(poisson_genes), " poisson genes")
            print("Setting there estimates to Inf")

        model_parameters.loc[poisson_genes, "theta"] = npy.inf

    if theta_regularization == "theta":
        model_parameters["od_factor"] = npy.log10(model_parameters["theta"])
    else:
        model_parameters["od_factor"] = npy.log10(
            1 + npy.power(10, genes_log10_gmean_step1) / model_parameters["theta"]
        )

    end = time.time()
    step1_time = npy.ceil(end - start)
    if verbosity:
        print("Step1 done. Took {} seconds.".format(npy.ceil(end - start)))
    # Step 2: Do regularization

    # Remove high disp genes
    # Not optimal
    # TODO: Fix
    if verbosity:
        print("Running Step2")
    start = time.time()
    model_parameters_to_return = model_parameters.copy()
    genes_log10_gmean_step1_to_return = genes_log10_gmean_step1.copy()
    genes_log10_amean_step1_to_return = genes_log10_amean_step1.copy()
    outliers_df = pd.DataFrame(index=genes_step1)
    for col in model_parameters.columns:
        col_outliers = is_outlier(model_parameters[col].values, genes_log10_gmean_step1)
        #col_outliers = is_outlier_r(
        #    model_parameters[col].values, genes_log10_gmean_step1
        #)
        outliers_df[col] = col_outliers
    non_outliers = outliers_df.sum(1) == 0
    non_outliers = non_outliers.tolist()
    outliers = outliers_df.sum(1) > 0
    outliers = outliers.tolist()
    if verbosity:
        print("outliers: {}".format(npy.sum(outliers)))

    genes_non_outliers = list(genes_step1[non_outliers])
    genes_step1 = list(genes_step1[non_outliers])
    genes_log10_gmean_step1 = genes_log10_gmean_step1[non_outliers]
    model_parameters = model_parameters.loc[genes_non_outliers]
    if method == "offset":
        model_parameters_fit = model_parameters.copy()
    else:
        model_parameters_fit = get_regularized_params(
            model_parameters,
            genes,
            genes_step1,
            genes_log10_gmean_step1,
            genes_log10_gmean,
            cell_attr,
            umi,
            theta_regularization=theta_regularization,
            exclude_poisson=exclude_poisson,
            poisson_genes=poisson_genes,
        )
    end = time.time()
    step2_time = npy.ceil(end - start)
    if verbosity:
        print("Step2 done. Took {} seconds.".format(npy.ceil(end - start)))

    # Step 3: Calculate residuals
    if verbosity:
        print("Running Step3")

    start = time.time()
    residuals = pd.DataFrame(
        get_residuals(umi, model_matrix, model_parameters_fit, residual_type)
    )
    residuals.index = genes
    residuals.columns = cell_names
    end = time.time()
    step3_time = npy.ceil(end - start)
    if verbosity:
        print("Step3 done. Took {} seconds.".format(npy.ceil(end - start)))

    gene_attr["theta_regularized"] = model_parameters["theta"]
    gene_attr["residual_mean"] = residuals.mean(1)
    gene_attr["residual_variance"] = residuals.var(1)

    corrected_counts = correct(
        residuals, cell_attr, latent_var, model_parameters_fit, umi
    )

    return {
        "residuals": residuals,
        "model_parameters": model_parameters_to_return,
        "model_parameters_fit": model_parameters_fit,
        "corrected_counts": corrected_counts,
        "genes_log10_gmean_step1": genes_log10_gmean_step1_to_return,
        "genes_log10_gmean": genes_log10_gmean,
        "genes_log10_amean_step1": genes_log10_amean_step1_to_return,
        "genes_log10_amean": genes_log10_amean,
        "cell_attr": cell_attr,
        "model_matrix": model_matrix,
        "gene_attr": gene_attr,
        "step1_time": step1_time,
        "step2_time": step2_time,
        "step3_time": step3_time,
    }
