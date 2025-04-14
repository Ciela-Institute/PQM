import numpy as np
import torch

from pqm import pqm
import pytest


def _get_dist(dist, size):
    if dist == "normal":
        return np.random.normal(size=size)
    elif dist == "cauchy":
        return np.random.standard_cauchy(size=size)
    else:
        raise ValueError("Invalid dist")


@pytest.mark.parametrize("return_type", ["p_value", "chi2"])
@pytest.mark.parametrize("dist", ["normal", "cauchy"])
@pytest.mark.parametrize("use_pytorch", [True, False])
def test_permute_fail(return_type, dist, use_pytorch):
    x_samples = _get_dist(dist, (256, 2))
    y_samples = _get_dist(dist, (256, 2))
    x_samples += 1 if dist == "normal" else 10

    if use_pytorch:
        x_samples = torch.tensor(x_samples)
        y_samples = torch.tensor(y_samples)

    base, permute = pqm(
        x_samples, y_samples, permute_tests=100, re_tessellation=100, return_type=return_type
    )
    if return_type == "p_value":
        pval = np.mean(np.mean(permute, axis=1) < np.mean(base))
    else:
        pval = np.mean(np.mean(permute, axis=1) > np.mean(base))

    assert pval < 5e-2
    assert np.all(np.isfinite(base))
    assert np.all(np.isfinite(permute))
