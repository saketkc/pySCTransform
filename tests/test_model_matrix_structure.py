# In tests/test_model_formula.py

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from pysctransform.pysctransform import build_model_formula


@pytest.fixture
def two_batch_data():
    return pd.DataFrame(
        {
            'log10_umi': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'batch': ['A', 'A', 'A', 'B', 'B', 'B']
        },
    )


@pytest.fixture
def three_batch_data():
    return pd.DataFrame(
        {
            'log10_umi': [1.0, 2.0, 3.0],
            'batch': ['X', 'Y', 'Z']
        },
    )


class TestModelMatrixStructure:

    def test_no_batch_has_intercept_and_slope(self, two_batch_data):
        formula = build_model_formula(['log10_umi'], batch_var=None)
        mm = dmatrix(formula, two_batch_data)

        columns = mm.design_info.column_names
        assert 'Intercept' in columns
        assert 'log10_umi' in columns
        assert len(columns) == 2

    @pytest.mark.parametrize(
        "fixture_name,batch_levels,rows_per_batch", [
            ("two_batch_data", ['A', 'B'], 3),
            ("three_batch_data", ['X', 'Y', 'Z'], 1),
        ],
    )
    def test_with_batch_structure(
            self, fixture_name, batch_levels, rows_per_batch, request,
    ):
        data = request.getfixturevalue(fixture_name)
        formula = build_model_formula(['log10_umi'], batch_var='batch')
        mm = dmatrix(formula, data)
        mm_arr = np.asarray(mm)
        columns = mm.design_info.column_names

        n_batches = len(batch_levels)

        # No global intercept
        assert 'Intercept' not in columns

        # Correct number of columns: n_batches * (1 intercept + 1 slope)
        assert len(columns) == n_batches * 2

        # Check each batch has its own intercept and slope columns
        for batch in batch_levels:
            intercept_cols = [i for i, c in enumerate(columns)
                              if f'[{batch}]' in c and 'log10_umi' not in c]
            slope_cols = [i for i, c in enumerate(columns)
                          if f'[{batch}]' in c and 'log10_umi' in c]

            assert len(
                intercept_cols,
            ) == 1, f"Expected 1 intercept column for batch {batch}"
            assert len(slope_cols) == 1, f"Expected 1 slope column for batch {batch}"

        # Check mutual exclusivity: each batch's rows have zeros for other batch columns
        for i, batch in enumerate(batch_levels):
            row_start = i * rows_per_batch
            row_end = row_start + rows_per_batch

            other_batches = [b for b in batch_levels if b != batch]
            for other in other_batches:
                other_cols = [j for j, c in enumerate(columns) if f'[{other}]' in c]
                assert (mm_arr[row_start:row_end, other_cols] == 0).all(), \
                    f"Batch {batch} rows should have zeros in batch {other} columns"

            # Intercept column should be 1 for this batch's rows
            intercept_idx = [j for j, c in enumerate(columns)
                             if f'[{batch}]' in c and 'log10_umi' not in c][0]
            assert (mm_arr[row_start:row_end, intercept_idx] == 1).all(), \
                f"Batch {batch} intercept should be 1 for its rows"
