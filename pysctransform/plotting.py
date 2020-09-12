# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def is_outlier(x, snr_threshold=25):
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


def plot_fit(pysct_results, fig=None):
    """
    Parameters
    ----------
    pysct_results: dict
                   obsect returned by pysctransform.vst
    """
    if fig is None:
        fig = plt.figure(figsize=(9, 3))
    genes_log10_gmean = pysct_results["genes_log10_gmean"]
    model_params = pysct_results["model_parameters"]
    model_params_fit = pysct_results["model_parameters_fit"]

    total_params = model_params_fit.shape[1]

    for index, column in enumerate(model_params_fit.columns):
        ax = fig.add_subplot(1, total_params, index + 1)
        model_param_col = model_params[column]
        model_param_outliers = is_outlier(model_param_col)
        ax.scatter(
            genes_log10_gmean[~model_param_outliers],
            model_param_col[~model_param_outliers],
            s=1,
            label="single gene estimate",
            color="#2b8cbe",
        )
        ax.scatter(
            genes_log10_gmean,
            model_params_fit[column],
            s=2,
            label="regularized",
            color="#de2d26",
        )
        ax.set_xlabel("log10(gene_gmean)")
        ax.set_ylabel(column)
        ax.set_title(column)
        ax.legend(frameon=False)
    _ = fig.tight_layout()
    return fig
