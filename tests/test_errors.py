import pytest
from pqm import pqm_pvalue, pqm_chi2
import numpy as np


def test_num_refs():
    with pytest.raises(ValueError):
        pqm_pvalue(np.random.normal(size=(10, 50)), np.random.normal(size=(15, 50)), num_refs=100)
    with pytest.raises(ValueError):
        pqm_chi2(np.random.normal(size=(15, 50)), np.random.normal(size=(10, 50)), num_refs=100)

    with pytest.warns(UserWarning):
        pqm_pvalue(
            np.random.normal(size=(100, 100)), np.random.normal(size=(110, 100)), num_refs=150
        )


def test_filled_bins():
    with pytest.raises(ValueError):
        pqm_pvalue(np.zeros(shape=(500, 50)), np.zeros(shape=(250, 50)), num_refs=10)
