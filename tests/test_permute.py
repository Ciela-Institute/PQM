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
def test_permute_fail(dist, use_pytorch):
    x_samples = _get_dist(dist, (256, 2))
    y_samples = _get_dist(dist, (256, 2))
    x_samples += 1 if dist == "normal" else 10

    if use_pytorch:
        x_samples = torch.tensor(x_samples)
        y_samples = torch.tensor(y_samples)

    base, permute = pqm_pvalue(x_samples, y_samples, permute_tests=32)
    pval = np.mean(np.array(permute) < base)
    assert pval < 5e-2
    assert np.all(np.isfinite(base))
    assert np.all(np.isfinite(permute))

    base, permute = pqm_chi2(x_samples, y_samples, permute_tests=32)
    pval = np.mean(np.array(permute) > base)
    assert pval < 5e-2
    assert np.all(np.isfinite(base))
    assert np.all(np.isfinite(permute))
