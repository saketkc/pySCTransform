import pytest
import pandas as pandas
from patsy import dmatrix
from pysctransform.pysctransform import (
    build_model_formula,
    make_cell_attr,
)
from tests.utils import (
    load_r_reference,
)

from tests.utils import (
    get_pbmc3k_filtered,
)


@pytest.fixture(scope="session")
def pbmc3k_data(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("pbmc3k_cache")
    return get_pbmc3k_filtered(cache_dir)


# conftest.py
@pytest.fixture(scope="session")
def batch_cell_attr():
    return pandas.read_csv("./data/r_batch_cell_attr.csv", index_col=0)


@pytest.fixture(scope="session")
def pbmc3k_batch_model(pbmc3k_data, batch_cell_attr):
    matrix, genes, cells = pbmc3k_data
    cell_attr = make_cell_attr(matrix, cells)
    cell_attr["batch"] = batch_cell_attr.loc[cell_attr.index, "batch"].values
    formula = build_model_formula(["log10_umi"], batch_var="batch")
    design_matrix = dmatrix(formula, cell_attr)
    return matrix, genes, cells, cell_attr, design_matrix, formula


@pytest.fixture(scope="session")
def r_batch_reference():
    return load_r_reference("./data/r_batch_model_pars.csv")


@pytest.fixture(scope="session")
def r_batch_fit_reference():
    return load_r_reference("./data/r_batch_model_pars_fit.csv")
