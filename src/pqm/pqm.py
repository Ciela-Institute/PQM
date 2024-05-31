from typing import Optional

import numpy as np
from scipy.stats import chi2_contingency
from scipy.spatial import KDTree

__all__ = ("pqm_chi2", "pqm_pvalue")

def _pqm_test(x_samples: np.ndarray, y_samples: np.ndarray, num_refs: int):
    """
    Helper function to perform the PQM test and return the results from chi2_contingency.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution, test samples.
    y_samples : np.ndarray
        Samples from the second distribution, reference samples.
    num_refs : int
        Number of reference samples to use.

    Returns
    -------
    tuple
        Results from scipy.stats.chi2_contingency function.
    """
    if len(y_samples) < num_refs:
        raise ValueError(
            "Number of reference samples must be less than the number of true samples."
        )
    elif len(y_samples) < 2 * num_refs:
        print(
            "Warning: Number of y_samples is small (less than twice the number of reference samples). Result may have high variance."
        )

    refs = np.random.choice(len(y_samples), num_refs, replace=False)
    N = np.arange(len(y_samples))
    N[refs] = -1
    N = N[N >= 0]
    refs, y_samples = y_samples[refs], y_samples[N]

    tree = KDTree(refs)

    idx = tree.query(x_samples, k=1, workers=-1)[1]
    counts_x = np.bincount(idx, minlength=num_refs)

    idx = tree.query(y_samples, k=1, workers=-1)[1]
    counts_y = np.bincount(idx, minlength=num_refs)

    # Remove reference samples with no counts
    C = (counts_x > 0) | (counts_y > 0)
    counts_x, counts_y = counts_x[C], counts_y[C]

    return chi2_contingency(np.array([counts_x, counts_y]))

def pqm_pvalue(x_samples: np.ndarray, y_samples: np.ndarray, num_refs: int = 100, bootstrap: Optional[int] = None):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples` are drawn form the same distribution.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution, test samples. Must have shape (N, *D) N is the number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution, reference samples. Must have shape (M, *D) M is the number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. Note that these will be drawn from y_samples, and then removed from the y_samples array.
    bootstrap : Optional[int]
        Number of bootstrap iterations to perform. No bootstrap if None (default).

    Returns
    -------
    float or list
        pvalue(s). Null hypothesis that both samples are drawn from the same distribution.
    """
    if bootstrap is not None:
        return [pqm_pvalue(x_samples, y_samples, num_refs=num_refs) for _ in range(bootstrap)]
    _, pvalue, _, _ = _pqm_test(x_samples, y_samples, num_refs)
    return pvalue

def pqm_chi2(x_samples: np.ndarray, y_samples: np.ndarray, num_refs: int = 100, bootstrap: Optional[int] = None):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples` are drawn form the same distribution.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution, test samples. Must have shape (N, *D) N is the number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution, reference samples. Must have shape (M, *D) M is the number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. Note that these will be drawn from y_samples, and then removed from the y_samples array.
    bootstrap : Optional[int]
        Number of bootstrap iterations to perform. No bootstrap if None (default).

    Returns
    -------
    float or list
        chi2 statistic(s) and degree(s) of freedom.
    """
    if bootstrap is not None:
        return [pqm_chi2(x_samples, y_samples, num_refs=num_refs) for _ in range(bootstrap)]
    chi2_stat, _, dof, _ = _pqm_test(x_samples, y_samples, num_refs)
    return chi2_stat, dof