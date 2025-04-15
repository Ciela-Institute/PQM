import numpy as np
from scipy.stats import chi2

from pqm import utils
import pytest


@pytest.mark.parametrize("dof", [2, 50, 100])
def test_max_test(dof):
    np.random.seed(42)
    c = chi2.ppf(np.random.uniform(size=10), df=dof)

    p = utils.max_test(c, dof)
    assert p > 1e-4 and p < 0.9999, "p-value out of range, expected U(0,1)"

    c[0] = 1000
    p = utils.max_test(c, dof)
    assert p < 1e-4, "p-value should be very small, expected ~0"
