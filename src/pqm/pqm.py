from typing import Optional

import numpy as np
from scipy.stats import chi2_contingency
from scipy.spatial import KDTree

__all__ = "pqm_pvalue"

def pqm_pvalue(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    num_refs: int = 100,
    bootstrap: Optional[int] = None,
    return_stat: str = "pvalue"

):
    """
    Perform the PQM test of the null hypothesis that `x_samples` and `y_samples` are drawn form the same distribution.

    Parameters
    ----------
    x_samples : np.ndarray
        Samples from the first distribution, samples from the posterior. Must have shape (N, *D) N is the number of x samples, and D is the dimensionality of the samples.
    y_samples : np.ndarray
        Samples from the second distribution, truth samples. Must have shape (M, *D) M is the number of y samples, and D is the dimensionality of the samples.
    num_refs : int
        Number of reference samples to use. Note that these will be drawn from y_samples, and then removed from the y_samples array.
    bootstrap : Optional[int]
        Number of bootstrap iterations to perform. No bootstrap if None (default).
    return_stat: str
        The statistic to return from chi2_contingency. Must be 'pvalue', 'chi2', or 'both'.

    Returns
    -------
    float
        pvalue. Null hypothesis that both samples are drawn from the same distribution.
    """
    if bootstrap is not None:
        return list(pqm_pvalue(x_samples, y_samples, num_refs=num_refs) for _ in range(bootstrap))
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

    chi2, pvalue, _, _ = chi2_contingency(np.array([counts_x, counts_y]))

    if return_stat == "pvalue":
        return pvalue
    elif return_stat == "chi2":
        return chi2
    elif return_stat == "both":
        return chi2, pvalue
    else:
        raise ValueError("Invalid value for return_stat. Must be 'pvalue', 'chi2', or 'both'.")
