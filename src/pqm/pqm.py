from typing import Optional
import warnings

import numpy as np
from scipy.stats import chi2_contingency, chi2
from scipy.spatial import KDTree

__all__ = ("pqm_chi2", "pqm_pvalue")


def _mean_std(sample1, sample2, axis=0):
    """Get the mean and std of two combined samples, without actually combining them, to save memory."""
    n1, *_ = sample1.shape
    n2, *_ = sample2.shape
    # Get mean/std of combined sample
    mx, sx = np.mean(sample1, axis=axis), np.std(sample1, axis=axis)
    my, sy = np.mean(sample2, axis=axis), np.std(sample2, axis=axis)
    m = (n1 * mx + n2 * my) / (n1 + n2)
    s = np.sqrt(
        ((n1 - 1) * (sx**2) + (n2 - 1) * (sy**2) + n1 * n2 * (mx - my) ** 2 / (n1 + n2))
        / (n1 + n2 - 1)
    )
    return m, s


def _pqm_test(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    num_refs: int,
    whiten: bool,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
):
    """
    Helper function to perform the PQM test and return the results from
    chi2_contingency.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution, test samples.
    y_samples : np.ndarray
        Samples from the second distribution, reference samples.
    num_refs : int
        Number of reference samples to use.
    whiten : bool
        If True, whiten the samples by subtracting the mean and dividing by the
        standard deviation.
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

    Returns
    -------
    tuple
        Results from scipy.stats.chi2_contingency function.
    """
    nx, *D = x_samples.shape
    ny, *D = y_samples.shape
    if (nx + ny) < num_refs:
        raise ValueError("Number of reference samples must be less than the number of samples.")
    elif (nx + ny) < 2 * num_refs:
        warnings.warn(
            "Number of samples is small (less than twice the number of reference samples). Result will have high variance and/or be non-discriminating."
        )
    if whiten:
        mean, std = _mean_std(x_samples, y_samples)
        y_samples = (y_samples - mean) / std
        x_samples = (x_samples - mean) / std

    # Determine fraction of x_samples to use as reference samples
    if x_frac is None:
        x_frac = nx / (nx + ny)

    # Determine number of samples from each distribution (x_samples, y_samples, gaussian)
    Nx, Ny, Ng = np.random.multinomial(
        num_refs,
        [x_frac * (1.0 - gauss_frac), (1.0 - x_frac) * (1.0 - gauss_frac), gauss_frac],
    )
    assert (Nx + Ny + Ng) == num_refs, f"Something went wrong. Nx={Nx}, Ny={Ny}, Ng={Ng} should sum to num_refs={num_refs}"  # fmt: skip

    # Collect reference samples from x_samples
    xrefs = np.random.choice(nx, Nx, replace=False)
    xrefs, x_samples = x_samples[xrefs], np.delete(x_samples, xrefs, axis=0)

    # Collect reference samples from y_samples
    yrefs = np.random.choice(ny, Ny, replace=False)
    yrefs, y_samples = y_samples[yrefs], np.delete(y_samples, yrefs, axis=0)

    # Join the full set of reference samples
    refs = np.concatenate([xrefs, yrefs], axis=0)

    # get gaussian reference points if requested
    if Ng > 0:
        m, s = _mean_std(x_samples, y_samples)
        gauss_refs = np.random.normal(
            loc=m,
            scale=s,
            size=(Ng, *D),
        )
        refs = np.concatenate([refs, gauss_refs], axis=0)

    # Build KDtree to measure distances
    tree = KDTree(refs)

    idx = tree.query(x_samples, k=1, workers=-1)[1]
    counts_x = np.bincount(idx, minlength=num_refs)

    idx = tree.query(y_samples, k=1, workers=-1)[1]
    counts_y = np.bincount(idx, minlength=num_refs)

    # Remove reference samples with no counts
    C = (counts_x > 0) | (counts_y > 0)
    counts_x, counts_y = counts_x[C], counts_y[C]

    n_filled_bins = np.sum((counts_x + counts_y) > 0)
    if n_filled_bins == 1:
        raise ValueError(
            """
            Only one Voronoi cell has samples, so chi^2 cannot 
            be computed. This is likely due to a small number 
            of samples or a pathological distribution. If possible, 
            increase the number of x_samples and y_samples.
            """
        )
    if n_filled_bins < (num_refs // 4):
        warnings.warn(
            """
            Less than a quarter of the Voronoi cells have any samples in them.
            Possibly due to a small number of samples or a pathological
            distribution. Result may be unreliable. If possible, increase the
            number of x_samples and y_samples.
            """
        )

    return chi2_contingency(np.stack([counts_x, counts_y]))


def pqm_pvalue(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    num_refs: int = 100,
    re_tessellation: Optional[int] = None,
    whiten: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
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
    whiten : bool
        If True, whiten the samples by subtracting the mean and dividing by the
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
                whiten=whiten,
                x_frac=x_frac,
                gauss_frac=gauss_frac,
            )
            for _ in range(re_tessellation)
        ]
    _, pvalue, _, _ = _pqm_test(x_samples, y_samples, num_refs, whiten, x_frac, gauss_frac)
    return pvalue


def pqm_chi2(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    num_refs: int = 100,
    re_tessellation: Optional[int] = None,
    whiten: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
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
        Number of times pqm_pvalue is called, re-tesselating the space. No
        re_tessellation if None (default).
    whiten : bool
        If True, whiten the samples by subtracting the mean and dividing by the
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
                whiten=whiten,
                x_frac=x_frac,
                gauss_frac=gauss_frac,
            )
            for _ in range(re_tessellation)
        ]
    chi2_stat, _, dof, _ = _pqm_test(x_samples, y_samples, num_refs, whiten, x_frac, gauss_frac)

    if dof != num_refs - 1:
        # Rescale chi2 to new value which has the same cumulative probability
        if chi2_stat / dof < 10:
            cp = chi2.sf(chi2_stat, dof)
            chi2_stat = chi2.isf(cp, num_refs - 1)
        else:
            chi2_stat = chi2_stat * (num_refs - 1) / dof
        dof = num_refs - 1
    return chi2_stat
