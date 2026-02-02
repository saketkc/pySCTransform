from pysctransform.pysctransform import get_downsampling_params


class TestDownSample:
    def test_none_means_no_downsampling(self):
        downsample_cells, downsample_genes, n_cells, n_genes = (
            get_downsampling_params(
                None, None, 10000,
                5000,
            ))
        assert not downsample_cells
        assert not downsample_genes
        assert n_cells == 10000
        assert n_genes == 5000

    def test_fewer_means_downsampling(self):
        downsample_cells, downsample_genes, n_cells, n_genes = (
            get_downsampling_params(
                5000, 2000, 10000,
                5000,
            ))
        assert downsample_cells
        assert downsample_genes
        assert n_cells == 5000
        assert n_genes == 2000

    def test_more_means_clamp_to_max(self):
        downsample_cells, downsample_genes, n_cells, n_genes = (
            get_downsampling_params(
                20000, 10000, 10000,
                5000,
            ))
        assert not downsample_cells
        assert not downsample_genes
        assert n_cells == 10000
        assert n_genes == 5000

    def test_exact_means_no_downsample(self):
        downsample_cells, downsample_genes, _, _ = (
            get_downsampling_params(
                10000, 5000, 10000,
                5000,
            ))
        assert not downsample_cells
        assert not downsample_genes

    def test_one_exact_one_not_means_split(self):
        downsample_cells, downsample_genes, _, _ = (
            get_downsampling_params(
                5000, None, 10000,
                5000,
            ))
        assert downsample_cells
        assert not downsample_genes
