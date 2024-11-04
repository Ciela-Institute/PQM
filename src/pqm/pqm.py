from typing import Optional
import warnings
import torch
import numpy as np
from scipy.stats import chi2_contingency, chi2
from scipy.spatial.distance import cdist
from torch.distributions import Multinomial
from typing import Optional, Union, Tuple

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
    if device.type == 'cuda':
        # Move tensors to CPU and convert to NumPy
        chi2_stat_cpu = chi2_stat.cpu().item()  # Convert to float
        orig_dof_cpu = orig_dof.cpu().item()    # Convert to float
    else:
        chi2_stat_cpu = chi2_stat
        orig_dof_cpu = orig_dof

    if orig_dof_cpu == target_dof:
        return chi2_stat_cpu
        
    if chi2_stat_cpu / orig_dof_cpu < 10:
        # Use cumulative probability method for better accuracy
        cp = chi2.sf(chi2_stat_cpu, orig_dof_cpu)
        return chi2.isf(cp, target_dof)
    else:
        # Use simple scaling for large values
        return chi2_stat_cpu * target_dof / orig_dof_cpu        

def _chi2_contingency_torch(
    counts_x: torch.Tensor,
    counts_y: torch.Tensor
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Perform chi-squared contingency test using PyTorch tensors.
    
    Returns:
        chi2_stat (torch.Tensor): Chi-squared statistic.
        p_value (float): p-value.
        dof (torch.Tensor): Degrees of freedom.
        expected (torch.Tensor): Expected frequencies.
    """
    counts = torch.stack([counts_x, counts_y])
    
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
    O_masked = O[mask]
    E_masked = E[mask]
    
    # Compute chi-squared statistic
    chi2_stat = ((O_masked - E_masked) ** 2 / E_masked).sum()
    
    # Move dof and chi2_stat to the same device
    dof = torch.tensor(dof, dtype=torch.float32, device=chi2_stat.device)
    
    # Compute p-value using the survival function (1 - CDF)
    p_value = torch.special.gammaincc(dof / 2, chi2_stat / 2).item()
    
    return chi2_stat, p_value, dof, E

def _sample_reference_indices_numpy(Nx, nx, Ny, ny, Ng, x_samples, y_samples):
    """
    Helper function to sample references for CPU-based NumPy computations.

    Parameters
    ----------
    Nx : int
        Number of references to sample from x_samples.
    nx : int
        Number of samples in x_samples.
    Ny : int
        Number of references to sample from y_samples.
    ny : int
        Number of samples in y_samples.
    Ng : int
        Number of references to sample from a Gaussian distribution.

    Returns
    -------
    np.ndarray  
        References samples.
    """

    if Nx > nx:
        raise ValueError("Cannot sample more references from x_samples than available")
    if Ny > ny:
        raise ValueError("Cannot sample more references from y_samples than available")
    
    # Reference samples from x_samples
    xrefs_indices = np.random.choice(nx, Nx, replace=False)
    xrefs = x_samples[xrefs_indices]
    x_samples = np.delete(x_samples, xrefs_indices, axis=0)
    
    # Reference samples from y_samples
    yrefs_indices = np.random.choice(ny, Ny, replace=False)
    yrefs = y_samples[yrefs_indices]
    y_samples = np.delete(y_samples, yrefs_indices, axis=0)
    
    # Combine references
    refs = np.concatenate([xrefs, yrefs], axis=0)
    
    # Gaussian references
    if Ng > 0:
        m, s = _mean_std(x_samples, y_samples)
        gauss_refs = np.random.normal(
            loc=m,
            scale=s,
            size=(Ng, ) + tuple(x_samples.shape[1:])
        )
        refs = np.concatenate([refs, gauss_refs], axis=0)

    return refs

def _compute_distances_numpy(x_samples, y_samples, refs, current_num_refs, num_refs):
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
    distances_x = cdist(x_samples, refs, metric='euclidean')
    distances_y = cdist(y_samples, refs, metric='euclidean')
    
    # Nearest references
    idx_x = np.argmin(distances_x, axis=1)
    idx_y = np.argmin(distances_y, axis=1)
    
    # Counts
    counts_x = np.bincount(idx_x, minlength=current_num_refs)
    counts_y = np.bincount(idx_y, minlength=current_num_refs)
    
    # Remove references with no counts
    C = (counts_x > 0) | (counts_y > 0)
    counts_x = counts_x[C]
    counts_y = counts_y[C]
    
    n_filled_bins = np.sum(C)
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
    
    # Perform chi-squared test using SciPy
    contingency_table = np.stack([counts_x, counts_y])
    return chi2_contingency(contingency_table)

def _sample_reference_indices_torch(Nx, nx, Ny, ny, Ng, x_samples, y_samples, device):
    """
    Helper function to sample references for GPU-based Torch computations.

    Parameters
    ----------
    Nx : int
        Number of references to sample from x_samples.
    nx : int
        Number of samples in x_samples.
    Ny : int
        Number of references to sample from y_samples.
    ny : int
        Number of samples in y_samples.
    Ng : int
        Number of references to sample from a Gaussian distribution.

    Returns
    -------
    np.ndarray  
        References samples.
    """
    
    if Nx > nx:
        raise ValueError("Cannot sample more references from x_samples than available")
    if Ny > ny:
        raise ValueError("Cannot sample more references from y_samples than available")
    
    # Reference samples from x_samples
    x_indices = torch.randperm(nx, device=device)
    xrefs_indices = x_indices[:Nx]
    x_samples_indices = x_indices[Nx:]
    xrefs = x_samples[xrefs_indices]
    x_samples = x_samples[x_samples_indices]
    
    # Reference samples from y_samples
    y_indices = torch.randperm(ny, device=device)
    yrefs_indices = y_indices[:Ny]
    y_samples_indices = y_indices[Ny:]
    yrefs = y_samples[yrefs_indices]
    y_samples = y_samples[y_samples_indices]
    
    # Combine references
    refs = torch.cat([xrefs, yrefs], dim=0)
    
    # Gaussian references
    if Ng > 0:
        m, s = _mean_std(x_samples, y_samples)
        # Ensure m and s have the correct shape
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        gauss_refs = torch.normal(
            mean=m.repeat(Ng, 1),
            std=s.repeat(Ng, 1),
        )
        refs = torch.cat([refs, gauss_refs], dim=0)
    return refs
    
def _compute_distances_torch(x_samples, y_samples, refs, current_num_refs, num_refs):
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
    idx_x = distances_x.argmin(dim=1)
    counts_x = torch.bincount(idx_x, minlength=current_num_refs)
    
    distances_y = torch.cdist(y_samples, refs)
    idx_y = distances_y.argmin(dim=1)
    counts_y = torch.bincount(idx_y, minlength=current_num_refs)
    
    # Remove references with no counts
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
    
    # Perform chi-squared test using the PyTorch implementation
    chi2_stat, p_value, dof, expected = _chi2_contingency_torch(counts_x, counts_y)
    return chi2_stat, p_value, dof, expected

def _pqm_test(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int,
    z_score_norm: bool,
    x_frac: Optional[float],
    gauss_frac: float,
    device: str = torch.device("cpu"),
) -> Tuple:
    """
    Helper function to perform the PQM test and return the results from
    chi2_contingency (using SciPy or a PyTorch implementation).

    Parameters
    ----------
    y_samples : np.ndarray or torch.Tensor
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray or torch.Tensor
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
        Results from scipy.stats.chi2_contingency or the PyTorch implementation.
    """

    # Determine if we're working with NumPy or PyTorch
    is_numpy = isinstance(x_samples, np.ndarray) and isinstance(y_samples, np.ndarray)
    is_torch = isinstance(x_samples, torch.Tensor) and isinstance(y_samples, torch.Tensor)
    
    if not (is_numpy or is_torch):
        raise TypeError("x_samples and y_samples must both be either NumPy arrays or PyTorch tensors.")
    
    # Validate sample sizes
    nx = x_samples.shape[0]
    ny = y_samples.shape[0]
    if (nx + ny) <= num_refs + 2:
        raise ValueError(
            "Number of reference samples (num_refs) must be less than the number of x/y samples. Ideally much less."
        )
    elif (nx + ny) < 2 * num_refs:
        warnings.warn(
            "Number of samples is small (less than twice the number of reference samples). "
            "Result will have high variance and/or be non-discriminating."
        )
    
    # Z-score normalization
    if z_score_norm:
        mean, std = _mean_std(x_samples, y_samples)
        if is_numpy:
            x_samples = (x_samples - mean) / std
            y_samples = (y_samples - mean) / std
        elif is_torch:
            x_samples = (x_samples - mean) / std
            y_samples = (y_samples - mean) / std
    
    # Determine fraction of x_samples to use as reference samples
    if x_frac is None:
        x_frac = nx / (nx + ny)
    
    # Determine number of samples from each distribution
    if is_numpy:
        counts = np.random.multinomial(
            num_refs,
            [x_frac * (1.0 - gauss_frac), (1.0 - x_frac) * (1.0 - gauss_frac), gauss_frac],
        )
        Nx, Ny, Ng = counts
    elif is_torch:
        probs = torch.tensor(
            [
                x_frac * (1.0 - gauss_frac),
                (1.0 - x_frac) * (1.0 - gauss_frac),
                gauss_frac,
            ],
            device=device,
        )
        counts_tensor = Multinomial(total_count=num_refs, probs=probs).sample()
        counts = counts_tensor.round().long().cpu().numpy()
        Nx, Ny, Ng = counts.tolist()
    
    # Validate counts
    if Nx + Ny + Ng != num_refs:
        raise ValueError(
            f"Something went wrong. Nx={Nx}, Ny={Ny}, Ng={Ng} should sum to num_refs={num_refs}"
        )
    
    # Sampling reference indices
    if is_numpy:
        refs = _sample_reference_indices_numpy(Nx, nx, Ny, ny, Ng, x_samples, y_samples)
    elif is_torch:
        refs = _sample_reference_indices_torch(Nx, nx, Ny, ny, Ng, x_samples, y_samples, device)
    
    # Update num_refs in case Gaussian samples were added
    current_num_refs = refs.shape[0]
    
    # Compute nearest references and counts
    if is_numpy:
        return _compute_distances_numpy(x_samples, y_samples, refs, current_num_refs, num_refs)
    
    elif is_torch:
        return _compute_distances_torch(x_samples, y_samples, refs, current_num_refs, num_refs)

def pqm_pvalue(
    x_samples,
    y_samples,
    num_refs: int = 100,
    re_tessellation: Optional[int] = None,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
    device: str = torch.device("cpu"),
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
        Number of times _pqm_test is called, re-tesselating the space. No
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
        Device to use for computation. Default: 'cpu'.

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
    float or list
        pvalue(s). Null hypothesis that both samples are drawn from the same
        distribution.
    """
    # Check the device and convert to the respective type (Numpy or Torch) and call their respective _pqm_test function

    if device.type == 'cpu':
        # Check if x_samples and y_samples are not already NumPy arrays
        if not isinstance(x_samples, np.ndarray):
            x_samples = x_samples.cpu().numpy()
        if not isinstance(y_samples, np.ndarray):
            y_samples = y_samples.cpu().numpy()
    elif device.type == 'cuda':
        # Check if x_samples and y_samples are not already torch tensors
        if not torch.is_tensor(x_samples):
            x_samples = torch.tensor(x_samples, device=device)
        if not torch.is_tensor(y_samples):
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
    
    _, p_value, _, _ = _pqm_test(
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
    device: str = torch.device("cpu"),
):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples`
    are drawn from the same distribution. This version returns the chi^2
    statistic with dof = num_refs-1.

    Parameters
    ----------
    x_samples : np.ndarray or torch.Tensor
        Samples from the first distribution. Must have shape (N, *D) N is the
        number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray or torch.Tensor
        Samples from the second distribution. Must have shape (M, *D) M is the
        number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. These samples will be drawn from
        x_samples, y_samples, and/or a Gaussian distribution, see the note
        below.
    re_tessellation : Optional[int]
        Number of times _pqm_test is called, re-tesselating the space. No
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
        Device to use for computation. Default: 'cpu'.

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

    # Check the device and convert to the respective type (Numpy or Torch) and call their respective _pqm_test function
    if device.type == 'cpu':
        # Check if x_samples and y_samples are not already NumPy arrays
        if not isinstance(x_samples, np.ndarray):
            x_samples = x_samples.cpu().numpy()
        if not isinstance(y_samples, np.ndarray):
            y_samples = y_samples.cpu().numpy()
    elif device.type == 'cuda':
        # Check if x_samples and y_samples are not already torch tensors
        if not torch.is_tensor(x_samples):
            x_samples = torch.tensor(x_samples, device=device)
        if not torch.is_tensor(y_samples):
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