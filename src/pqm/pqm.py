from typing import Optional
import warnings
import torch
import numpy as np
from scipy.stats import chi2_contingency, chi2
from scipy.spatial import KDTree
from torch.distributions import Multinomial

__all__ = ("pqm_chi2", "pqm_pvalue")


def _mean_std(sample1, sample2, dim=0):
    """Get the mean and std of two combined samples without actually combining them."""
    n1 = sample1.shape[dim]
    n2 = sample2.shape[dim]
    # Get mean/std of combined sample
    mx = torch.mean(sample1, dim=dim)
    sx = torch.std(sample1, dim=dim, unbiased=True)
    my = torch.mean(sample2, dim=dim)
    sy = torch.std(sample2, dim=dim, unbiased=True)
    m = (n1 * mx + n2 * my) / (n1 + n2)
    s = torch.sqrt(
        (
            (n1 - 1) * (sx ** 2)
            + (n2 - 1) * (sy ** 2)
            + n1 * n2 * (mx - my) ** 2 / (n1 + n2)
        )
        / (n1 + n2 - 1)
    )
    return m, s


def rescale_chi2(chi2_stat, orig_dof, target_dof, device):
    """
    Rescale chi2 statistic using appropriate methods depending on the device.
    """        

    # Move tensors to CPU and convert to NumPy
    chi2_stat_cpu = chi2_stat.cpu().item()  # Convert to float
    orig_dof_cpu = orig_dof.cpu().item()    # Convert to float

    if orig_dof_cpu == target_dof:
        return chi2_stat_cpu
        
    if chi2_stat_cpu / orig_dof_cpu < 10:
        # Use cumulative probability method for better accuracy
        cp = chi2.sf(chi2_stat_cpu, orig_dof_cpu)
        return chi2.isf(cp, target_dof)
    else:
        # Use simple scaling for large values
        return chi2_stat_cpu * target_dof / orig_dof_cpu



def _chi2_contingency(counts, device):
    """
    Computes the chi-squared statistic and p-value for a contingency table.

    Parameters
    ----------
    counts: torch.Tensor
        2xN tensor of counts for each category.
    device : str
        Device to use for computation. Default: 'cpu'. If 'cuda' is selected,

    Returns
    -------
    tuple
        chi2_stat, p_value, dof, expected
    """
    if device == 'cpu':
        counts_np = counts.cpu().numpy()
        chi2_stat, p_value, dof, expected = chi2_contingency(counts_np)
        chi2_stat = torch.tensor(chi2_stat, device=device)
        dof = torch.tensor(dof, device=device)
        return chi2_stat, p_value, dof, expected
    else:
        # Observed counts
        O = counts.float()

        # Row sums and column sums
        row_sums = O.sum(dim=1, keepdim=True)  # shape (2, 1)
        col_sums = O.sum(dim=0, keepdim=True)  # shape (1, N)
        total = O.sum()

        # Expected counts under the null hypothesis of independence
        E = row_sums @ col_sums / total  # shape (2, N)

        # Degrees of freedom
        dof = (O.size(0) - 1) * (O.size(1) - 1)

        # Avoid division by zero
        mask = E > 0
        O = O[mask]
        E = E[mask]

        # Compute chi-squared statistic
        chi2_stat = ((O - E) ** 2 / E).sum()

        # Move dof and chi2_stat to the same device
        dof = torch.tensor(dof, dtype=torch.float32, device=chi2_stat.device)

        # Compute p-value using the survival function (1 - CDF)
        p_value = torch.special.gammaincc(dof / 2, chi2_stat / 2).item()

        return chi2_stat, p_value, dof, E


def _pqm_test(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    num_refs: int,
    z_score_norm: bool,
    x_frac: Optional[float],
    gauss_frac: float,
    device: str,
):
    """
    Helper function to perform the PQM test and return the results from
    chi2_contingency.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. These samples will be drawn from
        x_samples, y_samples, and/or a Gaussian distribution, see the note
        below.
    re_tessellation : Optional[int]
        Number of times pqm_pvalue is called, re-tesselating the space. No
        re_tessellation if None (default).
    z_score_norm : bool
        If True, z_score_norm the samples by subtracting the mean and dividing by the
        standard deviation. mean and std are calculated from the combined
        x_samples and y_samples.
    x_frac : float
        Fraction of x_samples to use as reference samples. ``x_frac = 1`` will
        use only x_samples as reference samples, ``x_frac = 0`` will use only
        y_samples as reference samples. Ideally, ``x_frac = len(x_samples) /
        (len(x_samples) + len(y_samples))`` which is what is done for x_frac =
        None (default).
    gauss_frac : float
        Fraction of samples to take from gaussian distribution with mean/std
        determined from the combined x_samples/y_samples. This ensures full
        support of the reference samples if pathological behavior is expected.
        Default: 0.0 no gaussian samples.

    Note
    ----
        When using ``x_frac`` and ``gauss_frac``, note that the number of
        reference samples from the x_samples, y_samples, and Gaussian
        distribution will be determined by a multinomial distribution. This
        means that the actual number of reference samples from each distribution
        may not be exactly equal to the requested fractions, but will on average
        equal those numbers. The mean relative number of reference samples drawn
        from x_samples, y_samples, and Gaussian is ``Nx=x_frac*(1-gauss_frac)``,
        ``Ny=(1-x_frac)*(1-gauss_frac)``, and ``Ng=gauss_frac`` respectively.
        For best results, we suggest using a large number of re-tessellations,
        though this is our recommendation in any case.
    device : str
        Device to use for computation. Default: 'cpu'. If 'cuda' is selected,

    Note
    ----
        When using ``x_frac`` and ``gauss_frac``, note that the number of
        reference samples from the x_samples, y_samples, and Gaussian
        distribution will be determined by a multinomial distribution. This
        means that the actual number of reference samples from each distribution
        may not be exactly equal to the requested fractions, but will on average
        equal those numbers. The mean relative number of reference samples drawn
        from x_samples, y_samples, and Gaussian is ``Nx=x_frac*(1-gauss_frac)``,
        ``Ny=(1-x_frac)*(1-gauss_frac)``, and ``Ng=gauss_frac`` respectively.
        For best results, we suggest using a large number of re-tessellations,
        though this is our recommendation in any case.

    Returns
    -------
    tuple
        Results from scipy.stats.chi2_contingency function.
    """
    nx = x_samples.shape[0]
    ny = y_samples.shape[0]
    if (nx + ny) <= num_refs + 2:
        raise ValueError(
            "Number of reference samples (num_ref) must be less than the number of x/y samples. Ideally much less."
        )
    elif (nx + ny) < 2 * num_refs:
        warnings.warn(
            "Number of samples is small (less than twice the number of reference samples). Result will have high variance and/or be non-discriminating."
        )
    if z_score_norm:
        mean, std = _mean_std(x_samples, y_samples)
        y_samples = (y_samples - mean) / std
        x_samples = (x_samples - mean) / std

    # Determine fraction of x_samples to use as reference samples
    x_frac = nx / (nx + ny) if x_frac is None else x_frac

    # Determine number of samples from each distribution
    probs = torch.tensor(
        [
            x_frac * (1.0 - gauss_frac),
            (1.0 - x_frac) * (1.0 - gauss_frac),
            gauss_frac,
        ],
        device=device,
    )

    counts = Multinomial(total_count=num_refs, probs=probs).sample()
    counts = counts.round().long()
    Nx, Ny, Ng = counts.tolist()
    assert (Nx + Ny + Ng) == num_refs, (
        f"Something went wrong. Nx={Nx}, Ny={Ny}, Ng={Ng} should sum to num_refs={num_refs}"
    )

    # Collect reference samples from x_samples
    x_indices = torch.randperm(nx, device=device)
    if Nx > nx:
        raise ValueError("Cannot sample more references from x_samples than available")
    xrefs_indices = x_indices[:Nx]
    x_samples_indices = x_indices[Nx:]

    xrefs = x_samples[xrefs_indices]
    x_samples = x_samples[x_samples_indices]

    # Collect reference samples from y_samples
    y_indices = torch.randperm(ny, device=device)
    if Ny > ny:
        raise ValueError("Cannot sample more references from y_samples than available")
    yrefs_indices = y_indices[:Ny]
    y_samples_indices = y_indices[Ny:]

    yrefs = y_samples[yrefs_indices]
    y_samples = y_samples[y_samples_indices]

    # Join the full set of reference samples
    refs = torch.cat([xrefs, yrefs], dim=0)

    # Get gaussian reference points if requested
    if Ng > 0:
        m, s = _mean_std(x_samples, y_samples)
        gauss_refs = torch.normal(
            mean=m.repeat(Ng, 1),
            std=s.repeat(Ng, 1),
        )
        refs = torch.cat([refs, gauss_refs], dim=0)

    num_refs = refs.shape[0]

    # Compute nearest reference for x_samples
    distances = torch.cdist(x_samples, refs)
    idx = distances.argmin(dim=1)
    counts_x = torch.bincount(idx, minlength=num_refs)

    # Compute nearest reference for y_samples
    distances = torch.cdist(y_samples, refs)
    idx = distances.argmin(dim=1)
    counts_y = torch.bincount(idx, minlength=num_refs)

    # Remove reference samples with no counts
    C = (counts_x > 0) | (counts_y > 0)
    counts_x = counts_x[C]
    counts_y = counts_y[C]

    n_filled_bins = C.sum().item()
    if n_filled_bins == 1:
        raise ValueError(
            """
            Only one Voronoi cell has samples, so chi^2 cannot 
            be computed. This is likely due to a small number 
            of samples or a pathological distribution. If possible, 
            increase the number of x_samples and y_samples.
            """
        )
    if n_filled_bins < (num_refs // 2):
        warnings.warn(
            """
            Less than half of the Voronoi cells have any samples in them.
            Possibly due to a small number of samples or a pathological
            distribution. Result may be unreliable. If possible, increase the
            number of x_samples and y_samples.
            """
        )

    # Perform chi-squared test
    counts = torch.stack([counts_x, counts_y])
    return _chi2_contingency(counts, device)


def pqm_pvalue(
    x_samples,
    y_samples,
    num_refs: int = 100,
    re_tessellation: Optional[int] = None,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
    device: str = 'cpu',
):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples`
    are drawn from the same distribution. This version returns the pvalue under
    the null hypothesis that both samples are drawn from the same distribution.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. These samples will be drawn from
        x_samples, y_samples, and/or a Gaussian distribution, see the note
        below.
    re_tessellation : Optional[int]
        Number of times pqm_pvalue is called, re-tesselating the space. No
        re_tessellation if None (default).
    z_score_norm : bool
        If True, z_score_norm the samples by subtracting the mean and dividing by the
        standard deviation. mean and std are calculated from the combined
        x_samples and y_samples.
    x_frac : float
        Fraction of x_samples to use as reference samples. ``x_frac = 1`` will
        use only x_samples as reference samples, ``x_frac = 0`` will use only
        y_samples as reference samples. Ideally, ``x_frac = len(x_samples) /
        (len(x_samples) + len(y_samples))`` which is what is done for x_frac =
        None (default).
    gauss_frac : float
        Fraction of samples to take from gaussian distribution with mean/std
        determined from the combined x_samples/y_samples. This ensures full
        support of the reference samples if pathological behavior is expected.
        Default: 0.0 no gaussian samples.

    Note
    ----
        When using ``x_frac`` and ``gauss_frac``, note that the number of
        reference samples from the x_samples, y_samples, and Gaussian
        distribution will be determined by a multinomial distribution. This
        means that the actual number of reference samples from each distribution
        may not be exactly equal to the requested fractions, but will on average
        equal those numbers. The mean relative number of reference samples drawn
        from x_samples, y_samples, and Gaussian is ``Nx=x_frac*(1-gauss_frac)``,
        ``Ny=(1-x_frac)*(1-gauss_frac)``, and ``Ng=gauss_frac`` respectively.
        For best results, we suggest using a large number of re-tessellations,
        though this is our recommendation in any case.
    device : str
        Device to use for computation. Default: 'cpu'. If 'cuda' is selected,

    Returns
    -------
    float or list
        pvalue(s). Null hypothesis that both samples are drawn from the same
        distribution.
    """
    # Move samples to torch tensors on the selected device
    x_samples = torch.tensor(x_samples, device=device)
    y_samples = torch.tensor(y_samples, device=device)

    if re_tessellation is not None:
        return [
            pqm_pvalue(
                x_samples,
                y_samples,
                num_refs=num_refs,
                z_score_norm=z_score_norm,
                x_frac=x_frac,
                gauss_frac=gauss_frac,
                device=device,
            )
            for _ in range(re_tessellation)
        ]
    chi2_stat, p_value, dof, _ = _pqm_test(
        x_samples, y_samples, num_refs, z_score_norm, x_frac, gauss_frac, device
    )
    
    # Return p-value as a float
    return p_value if isinstance(p_value, float) else float(p_value)


def pqm_chi2(
    x_samples,
    y_samples,
    num_refs: int = 100,
    re_tessellation: Optional[int] = None,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
    device: str = 'cpu',
):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples`
    are drawn from the same distribution. This version returns the chi^2
    statistic with dof = num_refs-1.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. These samples will be drawn from
        x_samples, y_samples, and/or a Gaussian distribution, see the note
        below.
    re_tessellation : Optional[int]
        Number of times pqm_chi2 is called, re-tesselating the space. No
        re_tessellation if None (default).
    z_score_norm : bool
        If True, z_score_norm the samples by subtracting the mean and dividing by the
        standard deviation. mean and std are calculated from the combined
        x_samples and y_samples.
    x_frac : float
        Fraction of x_samples to use as reference samples. ``x_frac = 1`` will
        use only x_samples as reference samples, ``x_frac = 0`` will use only
        y_samples as reference samples. Ideally, ``x_frac = len(x_samples) /
        (len(x_samples) + len(y_samples))`` which is what is done for x_frac =
        None (default).
    gauss_frac : float
        Fraction of samples to take from gaussian distribution with mean/std
        determined from the combined x_samples/y_samples. This ensures full
        support of the reference samples if pathological behavior is expected.
        Default: 0.0 no gaussian samples.
    device : str
        Device to use for computation. Default: 'cpu'. If 'cuda' is selected,

    Note
    ----
        When using ``x_frac`` and ``gauss_frac``, note that the number of
        reference samples from the x_samples, y_samples, and Gaussian
        distribution will be determined by a multinomial distribution. This
        means that the actual number of reference samples from each distribution
        may not be exactly equal to the requested fractions, but will on average
        equal those numbers. The mean relative number of reference samples drawn
        from x_samples, y_samples, and Gaussian is ``Nx=x_frac*(1-gauss_frac)``,
        ``Ny=(1-x_frac)*(1-gauss_frac)``, and ``Ng=gauss_frac`` respectively.
        For best results, we suggest using a large number of re-tessellations,
        though this is our recommendation in any case.

    Note
    ----
        Some voronoi bins may be empty after counting. Due to the nature of
        ``scipy.stats.chi2_contingency`` we must first remove those voronoi
        cells, meaning that the dof of the chi^2 would change. To mitigate this
        effect, we rescale the chi^2 statistic to have the same cumulative
        probability as the desired chi^2 statistic. Thus, the returned chi^2 is
        always in reference to dof = num_refs-1. Thus users should not need to
        worry about this, but it is worth noting, please contact us if you
        notice unusual behavior.

    Returns
    -------
    float or list
        chi2 statistic(s).
    """
    # Move samples to torch tensors on the selected device
    x_samples = torch.tensor(x_samples, device=device)
    y_samples = torch.tensor(y_samples, device=device)

    if re_tessellation is not None:
        return [
            pqm_chi2(
                x_samples,
                y_samples,
                num_refs=num_refs,
                z_score_norm=z_score_norm,
                x_frac=x_frac,
                gauss_frac=gauss_frac,
                device=device,
            )
            for _ in range(re_tessellation)
        ]

    chi2_stat, _, dof, _ = _pqm_test(
        x_samples, y_samples, num_refs, z_score_norm, x_frac, gauss_frac, device
    )

    # Rescale chi2 statistic if necessary
    if dof != num_refs - 1:
        chi2_stat = rescale_chi2(chi2_stat, dof, num_refs - 1, device)

    # Return chi2_stat as a float
    return chi2_stat.item() if isinstance(chi2_stat, torch.Tensor) else float(chi2_stat)