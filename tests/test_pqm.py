import numpy as np
import torch

from pqm import pqm_pvalue, pqm_chi2
import pytest


def _get_dist(dist, size):
    if dist == "normal":
        return np.random.normal(size=size)
    elif dist == "cauchy":
        return np.random.standard_cauchy(size=size)
    else:
        raise ValueError("Invalid dist")


@pytest.mark.parametrize("dist", ["normal", "cauchy"])
@pytest.mark.parametrize("use_pytorch", [True, False])
@pytest.mark.parametrize("z_score_norm", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_pass_pvalue(dist, use_pytorch, z_score_norm, num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = _get_dist(dist, (500, ndim))
        x_samples = _get_dist(dist, (250, ndim))
        if use_pytorch:
            x_samples = torch.tensor(x_samples)
            y_samples = torch.tensor(y_samples)
        new.append(pqm_pvalue(x_samples, y_samples, z_score_norm=z_score_norm, num_refs=num_refs))

    # Check for roughly uniform distribution of p-values
    assert np.abs(np.mean(new) - 0.5) < 0.15


@pytest.mark.parametrize("dist", ["normal", "cauchy"])
@pytest.mark.parametrize("use_pytorch", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_pass_chi2(dist, use_pytorch, num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = _get_dist(dist, (500, ndim))
        x_samples = _get_dist(dist, (250, ndim))
        if use_pytorch:
            x_samples = torch.tensor(x_samples)
            y_samples = torch.tensor(y_samples)

        new.append(pqm_chi2(x_samples, y_samples, num_refs=num_refs))
    new = np.array(new)
    assert np.abs(np.mean(new) / (num_refs - 1) - 1) < 0.15


@pytest.mark.parametrize("dist", ["normal", "cauchy"])
@pytest.mark.parametrize("use_pytorch", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_fail_pvalue(dist, use_pytorch, num_refs, ndim):
    new = []
    for _ in range(50):
        y_samples = _get_dist(dist, (500, ndim))
        y_samples += 1 if dist == "normal" else 10
        x_samples = _get_dist(dist, (500, ndim))

        if use_pytorch:
            x_samples = torch.tensor(x_samples)
            y_samples = torch.tensor(y_samples)

        new.append(pqm_pvalue(x_samples, y_samples, num_refs=num_refs))

    assert np.mean(new) < 1e-3


@pytest.mark.parametrize("dist", ["normal", "cauchy"])
@pytest.mark.parametrize("use_pytorch", [True, False])
@pytest.mark.parametrize("z_score_norm", [True, False])
@pytest.mark.parametrize("num_refs", [20, 100])
@pytest.mark.parametrize("ndim", [1, 50])
def test_fail_chi2(dist, use_pytorch, z_score_norm, num_refs, ndim):
    new = []
    for _ in range(100):
        y_samples = _get_dist(dist, (500, ndim))
        y_samples += 1 if dist == "normal" else 10
        x_samples = _get_dist(dist, (500, ndim))

        if use_pytorch:
            x_samples = torch.tensor(x_samples)
            y_samples = torch.tensor(y_samples)

        new.append(pqm_chi2(x_samples, y_samples, z_score_norm=z_score_norm, num_refs=num_refs))
    new = np.array(new)
    assert (np.mean(new) / (num_refs - 1)) > 1.5


@pytest.mark.parametrize(
    "dist,gauss_frac", [["normal", 0.0], ["normal", 0.5], ["normal", 1.0], ["cauchy", 0.0]]
)
@pytest.mark.parametrize("use_pytorch", [True, False])
@pytest.mark.parametrize("x_frac", [None, 0.0, 0.5, 1.0])
def test_fracs(dist, use_pytorch, x_frac, gauss_frac):
    x_samples = _get_dist(dist, (500, 50))
    y_samples = _get_dist(dist, (500, 50))
    x_samples += 1 if dist == "normal" else 10

    if use_pytorch:
        x_samples = torch.tensor(x_samples)
        y_samples = torch.tensor(y_samples)

    pval = pqm_pvalue(x_samples, y_samples, x_frac=x_frac, gauss_frac=gauss_frac)
    assert pval < 1e-3
