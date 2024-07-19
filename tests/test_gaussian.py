import numpy as np
from pqm import pqm_pvalue, pqm_chi2


def test_pass_pvalue():
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, 50))
        x_samples = np.random.normal(size=(250, 50))

        new.append(pqm_pvalue(x_samples, y_samples))

    assert np.abs(np.mean(new) - 0.5) < 0.15


def test_pass_chi2():
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, 50))
        x_samples = np.random.normal(size=(250, 50))

        new.append(pqm_chi2(x_samples, y_samples, num_refs=100))
    new = np.array(new)
    print("np.abs(np.mean(new) / 99 - 1) < 0.15")
    assert np.abs(np.mean(new) / 99 - 1) < 0.15


def test_fail_pvalue():
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, 50))
        x_samples = np.random.normal(size=(250, 50)) + 0.5

        new.append(pqm_pvalue(x_samples, y_samples))

    assert np.mean(new) < 1e-3


def test_fail_chi2(num_refs = 50):
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, 50))
        x_samples = np.random.normal(size=(250, 50)) + 0.5

        new.append(pqm_chi2(x_samples, y_samples, num_refs=num_refs))
    new = np.array(new)
    assert np.mean(new) / num_refs-1 > 2
