from typing import Optional, Union, Literal
from warnings import warn

import torch
import numpy as np
from scipy.stats import chi2

from .pqm_core import (
    single_pqm_test,
    permute_retesselate_pqm_test,
    permute_retesselate_pqm_test_slow,
)

__all__ = ("pqm",)


def pqm(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int = 100,
    return_type: Literal["p_value", "chi2"] = "p_value",
    re_tessellation: Optional[int] = None,
    permute_tests: Optional[int] = None,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: float = 0.0,
    kernel: str = "euclidean",
):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples`
    are drawn from the same distribution. This version returns the p-value under
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
    permute_tests : Optional[int]
        Number of permutation tests to perform. If not None, will return a
        p-value, the p-value on the original x/y data, and the p-values on the
        permuted data.
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
    kernel : str or callable
        Kernel function to use for distance calculation. If a string, must be
        one of 'euclidean', 'cityblock', 'cosine', 'chebyshev', 'canberra', or
        'correlation' (see ``scipy.distance.cdist``). If a callable, must take
        two vectors and return a scalar, should also be commutative. This only
        works for numpy array inputs. Default: 'euclidean'.

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
        p-value(s). Null hypothesis that both samples are drawn from the same
        distribution.
    """

    if permute_tests is not None or re_tessellation is not None:
        if permute_tests is None:
            permute_tests = 0
        if re_tessellation is None:
            re_tessellation = 1
        if gauss_frac > 0:
            warn("A faster implementation is available for the Gaussian fraction of 0.0.")
            return permute_retesselate_pqm_test_slow(
                x_samples,
                y_samples,
                num_refs=num_refs,
                re_tessellation=re_tessellation,
                permute_tests=permute_tests,
                z_score_norm=z_score_norm,
                x_frac=x_frac,
                kernel=kernel,
                return_type=return_type,
            )
        return permute_retesselate_pqm_test(
            x_samples,
            y_samples,
            num_refs=num_refs,
            re_tessellation=re_tessellation,
            permute_tests=permute_tests,
            z_score_norm=z_score_norm,
            x_frac=x_frac,
            kernel=kernel,
            return_type=return_type,
        )

    p_value = single_pqm_test(
        x_samples,
        y_samples,
        num_refs=num_refs,
        z_score_norm=z_score_norm,
        x_frac=x_frac,
        gauss_frac=gauss_frac,
        kernel=kernel,
    )

    if return_type == "p_value":
        return p_value
    chi2_stat = chi2.isf(p_value, num_refs - 1)
    return chi2_stat
