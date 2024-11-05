from typing import Optional, Union, Tuple
import warnings

import torch
import numpy as np
from scipy.stats import chi2_contingency

from .utils import (
    _mean_std_numpy,
    _mean_std_torch,
    _sample_reference_indices_numpy,
    _sample_reference_indices_torch,
    _compute_counts_numpy,
    _compute_counts_torch,
    _rescale_chi2,
)

__all__ = ("pqm_chi2", "pqm_pvalue")


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
    assert type(x_samples) == type(
        y_samples
    ), f"x_samples and y_samples must be of the same type, not {type(x_samples)} and {type(y_samples)}"
    is_torch = isinstance(x_samples, torch.Tensor) and isinstance(y_samples, torch.Tensor)
    if not is_torch:
        x_samples = np.asarray(x_samples)
        y_samples = np.asarray(y_samples)

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
        if is_torch:
            mean, std = _mean_std_torch(x_samples, y_samples)
        else:
            mean, std = _mean_std_numpy(x_samples, y_samples)

        x_samples = (x_samples - mean) / std
        y_samples = (y_samples - mean) / std

    # Determine fraction of x_samples to use as reference samples
    if x_frac is None:
        x_frac = nx / (nx + ny)

    # Determine number of samples from each distribution
    counts = np.random.multinomial(
        num_refs,
        [x_frac * (1.0 - gauss_frac), (1.0 - x_frac) * (1.0 - gauss_frac), gauss_frac],
    )
    Nx, Ny, Ng = counts

    # Validate counts
    if Nx + Ny + Ng != num_refs:
        raise ValueError(
            f"Something went wrong. Nx={Nx}, Ny={Ny}, Ng={Ng} should sum to num_refs={num_refs}"
        )
    if Nx >= x_samples.shape[0] - 2:
        raise ValueError("Cannot sample more references from x_samples than available")
    if Ny >= y_samples.shape[0] - 2:
        raise ValueError("Cannot sample more references from y_samples than available")

    # count samples in each voronoi bin
    if is_torch:
        refs, x_samples, y_samples = _sample_reference_indices_torch(
            Nx, Ny, Ng, x_samples, y_samples, device
        )
        counts_x, counts_y = _compute_counts_torch(x_samples, y_samples, refs, num_refs)
    else:
        refs, x_samples, y_samples = _sample_reference_indices_numpy(
            Nx, Ny, Ng, x_samples, y_samples
        )
        counts_x, counts_y = _compute_counts_numpy(x_samples, y_samples, refs, num_refs)

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

    # Perform chi-squared test using SciPy
    contingency_table = np.stack([counts_x, counts_y])
    return chi2_contingency(contingency_table)


def pqm_pvalue(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
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
    return p_value


def pqm_chi2(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
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
        chi2_stat = _rescale_chi2(chi2_stat, dof, num_refs - 1)

    # Return chi2_stat as a float
    return chi2_stat
