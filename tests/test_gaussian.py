import numpy as np
from pqm import pqm_pvalue, pqm_chi2
import pytest


@pytest.mark.parametrize("z_score_norm", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_pass_pvalue(z_score_norm, num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = np.random.normal(size=(500, ndim))
        x_samples = np.random.normal(size=(250, ndim))

        new.append(pqm_pvalue(x_samples, y_samples, z_score_norm=z_score_norm, num_refs=num_refs))

    # Check for roughly uniform distribution of p-values
    assert np.abs(np.mean(new) - 0.5) < 0.15


@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_pass_chi2(num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = np.random.normal(size=(500, ndim))
        x_samples = np.random.normal(size=(250, ndim))

        new.append(pqm_chi2(x_samples, y_samples, num_refs=num_refs))
    new = np.array(new)
    assert np.abs(np.mean(new) / (num_refs - 1) - 1) < 0.15


@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_fail_pvalue(num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = np.random.normal(size=(500, ndim))
        y_samples[:, 0] += 5  # one dim off by 5sigma
        x_samples = np.random.normal(size=(250, ndim))

        new.append(pqm_pvalue(x_samples, y_samples, num_refs=num_refs))

    assert np.mean(new) < 1e-3


@pytest.mark.parametrize("z_score_norm", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_fail_chi2(z_score_norm, num_refs, ndim):
    new = []
    for _ in range(100):
        y_samples = np.random.normal(size=(500, ndim))
        y_samples[:, 0] += 5  # one dim off by 5sigma
        x_samples = np.random.normal(size=(250, ndim))

        new.append(pqm_chi2(x_samples, y_samples, z_score_norm=z_score_norm, num_refs=num_refs))
    new = np.array(new)
    assert (np.mean(new) / (num_refs - 1)) > 1.5


@pytest.mark.parametrize("x_frac", [None, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("gauss_frac", [0.0, 0.5, 1.0])
def test_fracs(x_frac, gauss_frac):
    x_samples = np.random.normal(size=(500, 50))
    y_samples = np.random.normal(size=(250, 50))
    x_samples[:, 0] += 5  # one dim off by 5sigma

    pval = pqm_pvalue(x_samples, y_samples, x_frac=x_frac, gauss_frac=gauss_frac)
    assert pval < 1e-3