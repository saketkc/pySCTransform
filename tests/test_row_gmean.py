import numpy as np
import scipy as scipy

from pysctransform.pysctransform import row_gmean  # adjust import


class TestRowGmean:
    def test_row_gmean_sparse_dense_equivalence(self):
        """Sparse and dense implementations should produce identical results."""
        np.random.seed(42)
        dense = np.random.rand(100, 500)
        dense[dense < 0.7] = 0  # Make it ~70% sparse
        sparse_mat = scipy.sparse.csr_matrix(dense)

        result_dense = row_gmean(dense)
        result_sparse = row_gmean(sparse_mat)

        np.testing.assert_allclose(result_dense, result_sparse, rtol=1e-10)
