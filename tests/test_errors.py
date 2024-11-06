import pytest
from pqm import pqm_pvalue, pqm_chi2
import numpy as np


def test_num_refs():
    # more num_refs than samples
    with pytest.raises(ValueError):
        pqm_pvalue(np.random.normal(size=(10, 50)), np.random.normal(size=(15, 50)), num_refs=100)

    # Not very many samples compared to num_refs
    with pytest.warns(UserWarning):
        pqm_pvalue(
            np.random.normal(size=(100, 100)), np.random.normal(size=(110, 100)), num_refs=150
        )

    # More x refs than x samples
    with pytest.raises(ValueError):
        pqm_pvalue(
            np.random.normal(size=(50, 100)),
            np.random.normal(size=(50, 100)),
            num_refs=50,
            x_frac=1.0,
        )
    # More y refs than y samples
    with pytest.raises(ValueError):
        pqm_pvalue(
            np.random.normal(size=(50, 100)),
            np.random.normal(size=(50, 100)),
            num_refs=50,
            x_frac=0.0,
        )


def test_filled_bins():
    with pytest.raises(ValueError):
        pqm_pvalue(np.zeros(shape=(500, 50)), np.zeros(shape=(250, 50)), num_refs=10)
