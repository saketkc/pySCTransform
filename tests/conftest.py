import pytest

from tests.utils import (
    get_pbmc3k_filtered,
)


@pytest.fixture(scope="session")
def pbmc3k_data(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("pbmc3k_cache")
    return get_pbmc3k_filtered(cache_dir)
