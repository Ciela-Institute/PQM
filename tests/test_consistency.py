from pqm.utils import _mean_std_torch, _mean_std_numpy, _compute_counts_numpy, _compute_counts_torch

import numpy as np
import torch

import pytest


@pytest.mark.parametrize("num_samples", [10, 100, 1000])
@pytest.mark.parametrize("dims", [10, 100])
def test_mean_std(num_samples, dims):
    x = np.random.rand(num_samples, dims)
    y = np.random.rand(num_samples, dims)
    m_np, s_np = _mean_std_numpy(x, y)
    assert m_np.shape == (dims,)
    assert s_np.shape == (dims,)

    x = torch.tensor(x)
    y = torch.tensor(y)
    m_pt, s_pt = _mean_std_torch(x, y)
    assert m_pt.shape == (dims,)
    assert s_pt.shape == (dims,)

    assert np.allclose(m_np, m_pt.numpy())
    assert np.allclose(s_np, s_pt.numpy())


@pytest.mark.parametrize("num_refs,num_samples", [[10, 100], [100, 1000]])
@pytest.mark.parametrize("dims", [10, 100])
def test_compute_counts(num_refs, num_samples, dims):

    x = np.random.rand(num_samples, dims)
    y = np.random.rand(num_samples, dims)
    refs = np.random.rand(num_refs, dims)

    counts_x_np, counts_y_np = _compute_counts_numpy(x, y, refs, num_refs)
    assert counts_x_np.shape == (num_refs,)

    x = torch.tensor(x)
    y = torch.tensor(y)
    refs = torch.tensor(refs)
    counts_x_pt, counts_y_pt = _compute_counts_torch(x, y, refs, num_refs)
    assert counts_x_pt.shape == (num_refs,)

    assert np.allclose(counts_x_np, counts_x_pt)
    assert np.allclose(counts_y_np, counts_y_pt)
