"""Main module."""
import warnings
import time
from KDEpy import FFTKDE
from scipy import interpolate
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import concurrent.futures
import logging

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from patsy import dmatrix
from statsmodels.api import GLM
from statsmodels.nonparametric.kernel_regression import KernelReg
from tqdm import tqdm

from scipy import stats

logging.captureWarnings(True)

from .fit import alpha_lbfgs
from .fit import theta_ml
from .fit import estimate_mu_glm
from .fit import estimate_mu_poisson


def is_outlier_naive(x, snr_threshold=25):
    """
    Mark points as outliers
    """
    if len(x.shape) == 1:
        x = x[:, None]
    median = np.median(x, axis=0)
    diff = np.sum((x - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > snr_threshold


def sparse_var(X, axis=None):
    X2 = X.copy()
    X2.data **= 2
    return X2.mean(axis) - np.square(X2.mean(axis))


def bwSJ(genes_log10_gmean_step1, bw_adjust=3):
    # See https://kdepy.readthedocs.io/en/latest/bandwidth.html
    fit = FFTKDE(kernel="gaussian", bw="ISJ").fit(genes_log10_gmean_step1)
    _ = fit.evaluate()
    bw = fit.bw * bw_adjust
    return np.array([bw], dtype=float)


def robust_scale(x):
    return (x - np.median(x)) / (
        stats.median_absolute_deviation(x) + np.finfo(float).eps
    )


def robust_scale_binned(y, x, breaks):
    # bins = pd.cut(x=x, bins=breaks, ordered=True)

    # categories = bins.categories
    bins = np.digitize(x=x, bins=breaks)
    categories = np.unique(bins)
    score = np.zeros(len(bins))
    for cat in categories:
        score_o = bins[bins == cat]
        score[bins == cat] = robust_scale(score_o)
    return score


def is_outlier(y, x, th=10):
    bin_width = (np.nanmax(x) - np.nanmin(x)) * bwSJ(x, bw_adjust=1 / 2)
    eps = np.finfo(float).eps * 10
    bin_width = bin_width[0]
    breaks1 = np.arange(
        start=np.nanmin(x) - eps, stop=np.nanmax(x) + bin_width, step=bin_width
    )
    breaks2 = np.arange(
        start=np.nanmin(x) - eps - bin_width / 2,
        stop=np.nanmax(x) + bin_width,
        step=bin_width,
    )
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return np.vstack((np.abs(score1), np.abs(score2))).min(0) > th


def make_cell_attr(umi, cell_names):
    assert umi.shape[1] == len(cell_names)
    total_umi = np.squeeze(np.asarray(umi.sum(0)))
    log10_umi = np.log10(total_umi)
    expressed_genes = np.squeeze(np.asarray((umi > 0).sum(0)))
    log10_expressed_genes = np.log10(expressed_genes)
    cell_attr = pd.DataFrame({"umi": total_umi, "log10_umi": log10_umi})
    cell_attr.index = cell_names
    cell_attr["n_expressed_genes"] = expressed_genes
    # this is referrred to as gene in SCTransform
    cell_attr["log10_gene"] = log10_expressed_genes
    cell_attr["umi_per_gene"] = log10_umi / expressed_genes
    cell_attr["log10_umi_per_gene"] = np.log10(cell_attr["umi_per_gene"])
    return cell_attr


def row_gmean(umi, gmean_eps=1):
    gmean = np.exp(np.log(umi + gmean_eps).mean(1)) - gmean_eps
    return gmean


def row_gmean_sparse(umi, gmean_eps=1):

    # print(umi.shape)
    # gmean = np.apply_along_axis(func1d=row_gmean, axis=1, arr=umi, gmean_eps=gmean_eps)
    # gmean = np.exp(np.log(umi + gmean_eps).mean(1)) - gmean_eps
    # for row in umi:
    gmean = np.asarray([row_gmean(x.todense(), gmean_eps)[0] for x in umi])
    gmean = np.squeeze(gmean)
    return gmean


def _process_y(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    y = np.asarray(y, dtype=int)
    y = np.squeeze(y)
    return y


def get_model_params_pergene(
    gene_umi, model_matrix, fit_type="theta_ml"
):  # latent_var, cell_attr):
    gene_umi = _process_y(gene_umi)
    if fit_type == "sm_nb":
        model = dm.NegativeBinomial(gene_umi, model_matrix, loglike_method="nb2")
        params = model.fit(maxiter=50, tol=1e-3, disp=0).params
        theta = 1 / params[-1]
        if theta >= 1e5:
            theta = np.inf
        params = dict(zip(model_matrix.design_info.column_names, params[:-1]))
        params["theta"] = theta
    elif fit_type == "theta_ml":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = theta_ml(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif fit_type == "alpha_lbfgs":
        params = estimate_mu_poisson(gene_umi, model_matrix)
        coef = params["coef"]
        mu = params["mu"]
        params = dict(zip(model_matrix.design_info.column_names, coef))
        theta = alpha_lbfgs(y=gene_umi, mu=mu)
        params["theta"] = theta
    elif fit_type == "tf":
        # pass
        params = fit_tensorflow(gene_umi, model_matrix)
    return params


def get_model_params_allgene(
    umi, model_matrix, fit_type="fit", threads=12, use_tf=False
):

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        if use_tf:
            feed_list = [
                (
                    tf.convert_to_tensor(row.values),
                    tf.convert_to_tensor(model_matrix),
                    # row.values,
                    # model_matrix,
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
            # TODO this should remain sparse
            # feed_list = [
            #    (row.values.reshape((-1, 1)), model_matrix, fit_type)
            #    for index, row in umi.iterrows()
            # ]
            feed_list = [
                (row.todense().reshape((-1, 1)), model_matrix, fit_type) for row in umi
            ]

            results = list(
                tqdm(
                    executor.map(lambda p: get_model_params_pergene(*p), feed_list),
                    total=len(feed_list),
                )
            )
    params_df = pd.DataFrame(results)

    return params_df


def dds(genes_log10_gmean_step1, grid_points=2 ** 10):
    # density dependent downsampling
    x, y = (
        FFTKDE(kernel="gaussian", bw="silverman")
        .fit(genes_log10_gmean_step1)
        .evaluate(grid_points=grid_points)
    )
    density = interpolate.interp1d(x=x, y=y, assume_sorted=False)
    sampling_prob = 1 / (density(genes_log10_gmean_step1) + np.finfo(float).eps)

    # sampling_prob = 1 / (density + np.finfo(float).eps)
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
    fixpoisson=False,
    poisson_genes=None,
):
    model_parameters = model_parameters.copy()
    genes_log10_gmean_step1 = genes_log10_gmean_step1[
        np.isfinite(model_parameters.theta)
    ]
    genes_step1 = genes_step1[np.isfinite(model_parameters.theta)]

    model_parameters_fit = pd.DataFrame(
        np.nan, index=genes, columns=model_parameters.columns
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

    x_points_df["x_points"] = np.nanmax(x_points_df, axis=1)
    x_points_df["max_gene_log10_gmean_step1"] = np.nanmax(genes_log10_gmean_step1)
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
        model_parameters_fit[column] = np.squeeze(fit[0])

    if theta_regularization == "theta":
        theta = np.power(10, (model_parameters["od_factor"]))
    else:
        theta = np.divide(
            np.power(10, genes_log10_gmean),
            np.power(10, model_parameters_fit["od_factor"] - 1),
            axis=0,
        )
    model_parameters_fit["theta"] = theta
    if fixpoisson:
        # relace theta by inf
        if poisson_genes is not None:
            model_parameters_fit.loc[poisson_genes, "theta"] = np.inf

    return model_parameters_fit


def get_residuals(umi, model_matrix, model_parameters_fit, res_clip_range="default"):

    subset = model_parameters_fit[model_matrix.design_info.column_names]
    theta = model_parameters_fit["theta"]

    mu = np.exp(subset.dot(model_matrix.T))
    # mu.columns = umi.columns

    variance = mu + (mu ** 2).divide(theta, axis=0)

    pearson_residuals = (umi - mu.values) / np.sqrt(variance.values)
    if res_clip_range == "default":
        res_clip_range = np.sqrt(umi.shape[1])
        pearson_residuals = np.clip(
            pearson_residuals, a_min=-res_clip_range, a_max=res_clip_range
        )
    return pearson_residuals


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
    fit_type="theta_ml",
    theta_regularization="od_factor",
    fixpoisson=False,
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
    cell_names = None
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

    gene_names = np.asarray(gene_names)
    cell_names = np.array(cell_names)

    if n_cells is None:
        n_cells = umi.shape[1]
    if n_genes is None:
        n_genes = umi.shape[0]

    genes_cell_count = np.asarray((umi > 0.01).sum(1))
    min_cells_genes_index = np.squeeze(genes_cell_count >= min_cells)
    genes = gene_names[min_cells_genes_index]
    if isinstance(umi, pd.DataFrame):
        umi = umi.loc[genes]
    else:
        umi = umi[min_cells_genes_index, :]
    genes_log10_gmean = np.log10(row_gmean_sparse(umi, gmean_eps=gmean_eps))

    # nonzerocells = cell_attr.loc[cell_attr.umi > 0].index
    # cell_attr = cell_attr.loc[nonzerocells]
    # umi = umi[nonzerocells]
    # n_cells = min(n_cells, umi.shape[1])
    """
    cells_step1 = np.random.choice(umi.columns.tolist(), size=n_cells, replace=False)
    genes_cell_count = (umi.loc[:, cells_step1] > 0).sum(1)
    genes = genes_cell_count >= min_cells
    umi = umi.loc[genes]
    genes_log10_gmean = np.log10(row_gmean(umi, gmean_eps))
    """

    cell_attr = make_cell_attr(umi, cell_names)

    if n_cells is None and n_cells < umi.shape[1]:
        # downsample cells to speed up the first step
        cells_step1_index = np.random.choice(
            a=np.arange(len(cell_names), dtype=int), size=n_cells, replace=False
        )
        cells_step1 = cell_names[cells_step1_index]
        genes_cell_count_step1 = (umi[:, cells_step1_index] > 0).sum(1)
        genes_step1 = genes[genes_cell_count_step1 >= min_cells]
        genes_log10_gmean_step1 = np.log10(
            row_gmean_sparse(umi[genes_step1, cells_step1], gmean_eps=gmean_eps)
        )
        umi_step1 = umi[:, cells_step1_index]
    else:
        cells_step1_index = np.arange(len(cell_names), dtype=int)
        cells_step1 = cell_names
        genes_step1 = genes
        genes_log10_gmean_step1 = genes_log10_gmean
        umi_step1 = umi

    data_step1 = cell_attr.loc[cells_step1]
    if (n_genes is not None) and (n_genes < len(genes_step1)):
        # density-sample genes to speed up the first step
        sampling_prob = dds(genes_log10_gmean_step1)

        genes_step1_index = np.random.choice(
            a=np.arange(len(genes_step1)), size=n_genes, replace=False, p=sampling_prob
        )
        genes_step1 = gene_names[genes_step1_index]
        umi_step1 = umi_step1[genes_step1_index, :]  # [:, cells_step1_index]
        genes_log10_gmean_step1 = np.log10(
            row_gmean_sparse(umi_step1, gmean_eps=gmean_eps)
        )

    # Step 1: Estimate theta
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

    model_parameters = get_model_params_allgene(
        umi_step1, model_matrix, fit_type, threads, use_tf
    )  # latent_var, cell_attr)
    model_parameters.index = genes_step1

    gene_attr = pd.DataFrame(index=genes)
    gene_attr["gene_amean"] = umi.mean(1)
    gene_attr["gene_gmean"] = np.power(10, genes_log10_gmean)
    gene_attr["gene_detectation_rate"] = (
        np.squeeze(np.asarray((umi > 0).sum(1))) / umi.shape[1]
    )
    gene_attr["theta"] = model_parameters["theta"]
    gene_attr["gene_variance"] = sparse_var(umi, 1)  # umi.var(1)

    poisson_genes = None
    if fixpoisson:
        poisson_genes = gene_attr[
            gene_attr["gene_amean"] >= gene_attr["gene_variance"]
        ].index.tolist()
        print("Found ", len(poisson_genes), " poisson genes")
        print("Setting there estimates to Inf")

        model_parameters.loc[poisson_genes, "theta"] = np.inf

    if theta_regularization == "theta":
        model_parameters["od_factor"] = np.log10(model_parameters["theta"])
    else:
        model_parameters["od_factor"] = np.log10(
            1
            + np.divide(
                np.power(10, genes_log10_gmean_step1), model_parameters["theta"], axis=0
            )
        )
    outliers_df = pd.DataFrame(index=genes_step1)
    for col in model_parameters.columns:
        col_outliers = is_outlier(model_parameters[col].values, genes_log10_gmean_step1)
        outliers_df[col] = col_outliers
    non_outliers = outliers_df.sum(1) == 0
    genes_non_outliers = genes_step1[non_outliers]
    model_parameters = model_parameters.loc[genes_non_outliers]

    end = time.time()
    print("Step1 done. Took {} seconds.".format(np.ceil(end - start)))
    # Step 2: Do regularization

    # Remove high disp genes
    # Not optimal
    # TODO: Fix
    print("Running Step2")
    start = time.time()
    model_parameters_fit = get_regularized_params(
        model_parameters,
        genes,
        genes_step1,
        genes_log10_gmean_step1,
        genes_log10_gmean,
        cell_attr,
        umi,
        theta_regularization=theta_regularization,
        fixpoisson=fixpoisson,
        poisson_genes=poisson_genes,
    )
    end = time.time()
    print("Step2 done. Took {} seconds.".format(np.ceil(end - start)))

    # Step 3: Calculate residuals
    print("Running Step3")
    start = time.time()
    residuals = pd.DataFrame(get_residuals(umi, model_matrix, model_parameters_fit))
    residuals.index = genes
    residuals.columns = cell_names
    end = time.time()
    print("Step3 done. Took {} seconds.".format(np.ceil(end - start)))

    gene_attr["theta_regularized"] = model_parameters["theta"]
    gene_attr["residual_mean"] = residuals.mean(1)
    gene_attr["residual_variance"] = residuals.var(1)

    return {
        "residuals": residuals,
        "model_parameters": model_parameters,
        "model_parameters_fit": model_parameters_fit,
        "genes_log10_gmean_step1": genes_log10_gmean_step1,
        "genes_log10_gmean": genes_log10_gmean,
        "cell_attr": cell_attr,
        "model_matrix": model_matrix,
        "gene_attr": gene_attr,
    }


def correct(pearson_residuals, cell_attr, latent_var, model_parameters_fit, umi):
    # replace value of latent variables with its median
    cell_attr = cell_attr.copy()
    for column in latent_var:
        cell_attr.loc[:, column] = cell_attr.loc[:, column].median()
    model_matrix = dmatrix(" + ".join(latent_var), cell_attr)
    non_theta_columns = [
        x for x in model_matrix.design_info.column_names if x != "theta"
    ]
    coefficients = model_parameters_fit[non_theta_columns]
    theta = model_parameters_fit["theta"]

    mu = np.exp(coefficients.dot(model_matrix.T))
    mu.columns = umi.columns

    variance = mu + (mu ** 2).divide(theta, axis=0)
    corrected_data = mu + pearson_residuals * np.sqrt(variance)
    corrected_data[corrected_data < 0] = 0
    corrected_data = corrected_data.astype(int)
    corrected_counts = pd.DataFrame(
        corrected_data, index=pearson_residuals.index, columns=umi.columns
    )
    return corrected_counts
