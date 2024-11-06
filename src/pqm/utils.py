import numpy as np
import torch
from scipy.stats import chi2
from scipy.spatial.distance import cdist

__all__ = (
    "_mean_std_torch",
    "_mean_std_numpy",
    "_rescale_chi2",
    "_chi2_contingency_torch",
    "_sample_reference_indices_numpy",
    "_compute_counts_numpy",
    "_sample_reference_indices_torch",
    "_compute_counts_torch",
)


def _mean_std_torch(sample1, sample2):
    """Get the mean and std of two combined samples without actually combining them."""
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]

    mx = torch.mean(sample1, dim=0)
    sx = torch.std(sample1, dim=0, unbiased=True)
    my = torch.mean(sample2, dim=0)
    sy = torch.std(sample2, dim=0, unbiased=True)

    m = (n1 * mx + n2 * my) / (n1 + n2)
    s = torch.sqrt(
        ((n1 - 1) * (sx**2) + (n2 - 1) * (sy**2) + n1 * n2 * (mx - my) ** 2 / (n1 + n2))
        / (n1 + n2 - 1)
    )
    return m, s


def _mean_std_numpy(sample1, sample2):
    """Get the mean and std of two combined samples without actually combining them."""
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]

    mx = np.mean(sample1, axis=0)
    sx = np.std(sample1, axis=0, ddof=1)
    my = np.mean(sample2, axis=0)
    sy = np.std(sample2, axis=0, ddof=1)

    m = (n1 * mx + n2 * my) / (n1 + n2)
    s = np.sqrt(
        ((n1 - 1) * (sx**2) + (n2 - 1) * (sy**2) + n1 * n2 * (mx - my) ** 2 / (n1 + n2))
        / (n1 + n2 - 1)
    )
    return m, s


def _rescale_chi2(chi2_stat, orig_dof, target_dof):
    """
    Rescale chi2 statistic using appropriate methods depending on the offset.
    """

    if chi2_stat / orig_dof < 10:
        # Use cumulative probability method for better accuracy
        cp = chi2.sf(chi2_stat, orig_dof)
        return chi2.isf(cp, target_dof)
    # Use simple scaling for large values
    return chi2_stat * target_dof / orig_dof


def _sample_reference_indices_torch(Nx, Ny, Ng, x_samples, y_samples, device):
    """
    Helper function to sample references for GPU-based Torch computations.

    Parameters
    ----------
    Nx : int
        Number of references to sample from x_samples.
    Ny : int
        Number of references to sample from y_samples.
    Ng : int
        Number of references to sample from a Gaussian distribution.

    Returns
    -------
    np.ndarray
        References samples.
    """

    # Reference samples from x_samples
    x_indices = torch.randperm(x_samples.shape[0], device=device)
    xrefs = x_samples[x_indices[:Nx]]
    x_samples = x_samples[x_indices[Nx:]]

    # Reference samples from y_samples
    y_indices = torch.randperm(y_samples.shape[0], device=device)
    yrefs = y_samples[y_indices[:Ny]]
    y_samples = y_samples[y_indices[Ny:]]

    # Combine references
    refs = torch.cat([xrefs, yrefs], dim=0)

    # Gaussian references
    if Ng > 0:
        m, s = _mean_std_torch(x_samples, y_samples)
        # Ensure m has the correct shape
        shaper = torch.ones(Ng, *x_samples.shape[1:], device=device)
        gauss_refs = torch.normal(mean=m * shaper, std=s)
        refs = torch.cat([refs, gauss_refs], dim=0)
    return refs, x_samples, y_samples


def _sample_reference_indices_numpy(Nx, Ny, Ng, x_samples, y_samples):
    """
    Helper function to sample references for CPU-based NumPy computations.

    Parameters
    ----------
    Nx : int
        Number of references to sample from x_samples.
    Ny : int
        Number of references to sample from y_samples.
    Ng : int
        Number of references to sample from a Gaussian distribution.

    Returns
    -------
    np.ndarray
        References samples.
    """

    # Reference samples from x_samples
    xrefs_indices = np.random.choice(x_samples.shape[0], Nx, replace=False)
    xrefs = x_samples[xrefs_indices]
    x_samples = np.delete(x_samples, xrefs_indices, axis=0)

    # Reference samples from y_samples
    yrefs_indices = np.random.choice(y_samples.shape[0], Ny, replace=False)
    yrefs = y_samples[yrefs_indices]
    y_samples = np.delete(y_samples, yrefs_indices, axis=0)

    # Combine references
    refs = np.concatenate([xrefs, yrefs], axis=0)

    # Gaussian references
    if Ng > 0:
        m, s = _mean_std_numpy(x_samples, y_samples)
        gauss_refs = np.random.normal(loc=m, scale=s, size=(Ng,) + tuple(x_samples.shape[1:]))
        refs = np.concatenate([refs, gauss_refs], axis=0)

    return refs, x_samples, y_samples


def _compute_counts_torch(x_samples, y_samples, refs, num_refs):
    """
    Helper function to calculate distances for GPU-based Torch computations.

    Parameters
    ----------
    x_samples : torch.Tensor
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : torch.Tensor
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    refs : torch.Tensor
        Reference samples. Must have shape (num_refs, *D) where D is the
        dimensionality of the samples.
    current_num_refs : int
        Number of reference samples used in the test.
    num_refs : int
        Number of reference samples to use.

    Returns
    -------
    tuple
        Results from the PyTorch implementation of chi2_contingency.
    """

    # Compute distances and find nearest references
    distances_x = torch.cdist(x_samples, refs)
    distances_y = torch.cdist(y_samples, refs)

    idx_x = distances_x.argmin(dim=1)
    idx_y = distances_y.argmin(dim=1)

    counts_x = torch.bincount(idx_x, minlength=num_refs)
    counts_y = torch.bincount(idx_y, minlength=num_refs)

    return counts_x.cpu().numpy(), counts_y.cpu().numpy()


def _compute_counts_numpy(x_samples, y_samples, refs, num_refs):
    """
    Helper function to calculate distances for CPU-based NumPy computations.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    refs : np.ndarray
        Reference samples. Must have shape (num_refs, *D) where D is the
        dimensionality of the samples.
    current_num_refs : int
        Number of reference samples used in the test.
    num_refs : int
        Number of reference samples to use.

    Returns
    -------
    tuple
        Results from scipy.stats.chi2_contingency.
    """

    # Compute distances
    distances_x = cdist(x_samples, refs, metric="euclidean")
    distances_y = cdist(y_samples, refs, metric="euclidean")

    # Nearest references
    idx_x = np.argmin(distances_x, axis=1)
    idx_y = np.argmin(distances_y, axis=1)

    # Counts
    counts_x = np.bincount(idx_x, minlength=num_refs)
    counts_y = np.bincount(idx_y, minlength=num_refs)

    return counts_x, counts_y
