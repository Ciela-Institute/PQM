import numpy as np
from .pqm import pqm_pvalue, pqm_chi2


def test():
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, 50))
        x_samples = np.random.normal(size=(250, 50))

        new.append(pqm_pvalue(x_samples, y_samples))

    assert np.abs(np.mean(new) - 0.5) < 0.15
