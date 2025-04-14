import pytest
from pqm import pqm
import numpy as np


@pytest.mark.parametrize("return_type", ["p_value", "chi2"])
def test_num_refs(return_type):
    # more num_refs than samples
    with pytest.raises(ValueError):
        pqm(
            np.random.normal(size=(10, 50)),
            np.random.normal(size=(15, 50)),
            num_refs=100,
            return_type=return_type,
        )

    # Not very many samples compared to num_refs
    with pytest.warns(UserWarning):
        pqm(
            np.random.normal(size=(100, 100)),
            np.random.normal(size=(110, 100)),
            num_refs=150,
            return_type=return_type,
        )

    # More x refs than x samples
    with pytest.raises(ValueError):
        pqm(
            np.random.normal(size=(50, 100)),
            np.random.normal(size=(50, 100)),
            num_refs=50,
            x_frac=1.0,
            return_type=return_type,
        )
    # More y refs than y samples
    with pytest.raises(ValueError):
        pqm(
            np.random.normal(size=(50, 100)),
            np.random.normal(size=(50, 100)),
            num_refs=50,
            x_frac=0.0,
            return_type=return_type,
        )


@pytest.mark.parametrize("return_type", ["p_value", "chi2"])
def test_filled_bins(return_type):
    with pytest.raises(ValueError):
        pqm(
            np.zeros(shape=(500, 50)),
            np.zeros(shape=(250, 50)),
            num_refs=10,
            return_type=return_type,
        )
