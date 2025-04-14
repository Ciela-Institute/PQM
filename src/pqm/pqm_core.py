from typing import Callable, Optional, Union

from scipy.stats import chi2_contingency, chi2
import numpy as np
from scipy.spatial.distance import cdist
import torch
from warnings import warn

from .utils import (
    _mean_std_numpy,
    _mean_std_torch,
    _sample_reference_indices_numpy,
    _sample_reference_indices_torch,
    _compute_counts_numpy,
    _compute_counts_torch,
    _random_permutation,
)


def init_checks_pqm_test(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
):
    assert type(x_samples) == type(
        y_samples
    ), f"x_samples and y_samples must be of the same type, not {type(x_samples)} and {type(y_samples)}"
    is_torch = isinstance(x_samples, torch.Tensor)
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
        warn(
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

    return x_samples, y_samples, x_frac, is_torch


def core_pqm_test(counts_x, counts_y):
    C = (counts_x > 0) | (counts_y > 0)
    counts_x = counts_x[C]
    counts_y = counts_y[C]

    n_filled_bins = C.sum().item()
    num_refs = len(C)
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
        warn(
            """
            Less than half of the Voronoi cells have any samples in them.
            Possibly due to a small number of samples or a pathological
            distribution. Result may be unreliable. If possible, increase the
            number of x_samples and y_samples.
            """
        )

    # Perform chi-squared test using SciPy
    contingency_table = np.stack([counts_x, counts_y])
    return chi2_contingency(contingency_table, correction=False)[1]  # p-value


def single_pqm_test(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int,
    return_type: str = "p_value",
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: Optional[float] = 0.0,
    kernel: Union[str, Callable] = "euclidean",
):
    x_samples, y_samples, x_frac, is_torch = init_checks_pqm_test(
        x_samples, y_samples, num_refs, z_score_norm, x_frac
    )

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
            Nx, Ny, Ng, x_samples, y_samples
        )
        counts_x, counts_y = _compute_counts_torch(x_samples, y_samples, refs, num_refs)
    else:
        refs, x_samples, y_samples = _sample_reference_indices_numpy(
            Nx, Ny, Ng, x_samples, y_samples
        )
        counts_x, counts_y = _compute_counts_numpy(x_samples, y_samples, refs, num_refs, kernel)

    p_value = core_pqm_test(counts_x, counts_y)
    if return_type == "p_value":
        return p_value
    elif return_type == "chi2":
        chi2_stat = chi2.isf(p_value, num_refs - 1)
        return chi2_stat
    else:
        raise ValueError(f"return_type must be 'p_value' or 'chi2', not {return_type}")


def permute_retesselate_pqm_test_slow(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int,
    re_tessellation: int = 100,
    permute_tests: int = 100,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    gauss_frac: Optional[float] = 0.0,
    kernel: Union[str, Callable] = "euclidean",
    return_type: str = "chi2",
):
    permute_stats = []
    for pt in range(permute_tests + 1):
        if pt > 0:
            x_samples, y_samples = _random_permutation(x_samples, y_samples)
        substats = []
        for _ in range(re_tessellation):
            value = single_pqm_test(
                x_samples,
                y_samples,
                num_refs,
                return_type=return_type,
                z_score_norm=z_score_norm,
                x_frac=x_frac,
                gauss_frac=gauss_frac,
                kernel=kernel,
            )
            substats.append(value)
        if re_tessellation == 1:
            substats = substats[0]
        if pt == 0:
            test_stat = substats
        else:
            permute_stats.append(substats)
    if permute_tests > 0:
        return np.array(test_stat), np.array(permute_stats)
    return np.array(test_stat)


def permute_retesselate_pqm_test(
    x_samples: Union[np.ndarray, torch.Tensor],
    y_samples: Union[np.ndarray, torch.Tensor],
    num_refs: int,
    re_tessellation: int = 100,
    permute_tests: int = 100,
    z_score_norm: bool = False,
    x_frac: Optional[float] = None,
    kernel: Union[str, Callable] = "euclidean",
    return_type: str = "chi2",
):
    x_samples, y_samples, x_frac, is_torch = init_checks_pqm_test(
        x_samples, y_samples, num_refs, z_score_norm, x_frac
    )
    nx = x_samples.shape[0]
    ny = y_samples.shape[0]
    p = np.concatenate((x_frac * np.ones(nx) / nx, (1 - x_frac) * np.ones(ny) / ny), axis=0)
    p /= p.sum()

    if is_torch:
        samples = torch.cat([x_samples, y_samples], dim=0)
        kernel = 2.0 if kernel == "euclidean" else kernel
        dmatrix = torch.cdist(samples, samples, p=kernel).detach().cpu().numpy()
    else:
        samples = np.concatenate([x_samples, y_samples], axis=0)
        dmatrix = cdist(samples, samples, metric=kernel)
    indices = np.arange(dmatrix.shape[0], dtype=np.int32)

    permute_stats = []
    for pt in range(permute_tests + 1):
        if pt > 0:
            np.random.shuffle(indices)
        substats = []
        for _ in range(re_tessellation):
            refs = np.random.choice(np.arange(dmatrix.shape[0]), size=num_refs, replace=False, p=p)
            subx = np.delete(indices[:nx], refs[refs < nx])
            idx = dmatrix[subx][:, indices[refs]].argmin(axis=1)
            counts_x = np.bincount(idx, minlength=num_refs)
            suby = np.delete(indices[nx:], refs[refs >= nx] - nx)
            idy = dmatrix[suby][:, indices[refs]].argmin(axis=1)
            counts_y = np.bincount(idy, minlength=num_refs)
            p_value = core_pqm_test(counts_x, counts_y)
            if return_type == "p_value":
                substats.append(p_value)
            else:
                chi2_stat = chi2.isf(p_value, num_refs - 1)
                substats.append(chi2_stat)
        if pt == 0:
            test_stat = substats
        else:
            permute_stats.append(substats)
    if permute_tests > 0:
        return np.array(test_stat), np.array(permute_stats)
    return np.array(test_stat)


if __name__ == "__main__":
    # Example usage
    x_samples = np.random.rand(128, 2000)
    y_samples = np.random.normal(size=(128, 2000))
    num_refs = 100
    from time import time

    start = time()
    res = permute_retesselate_pqm_test(
        x_samples,
        y_samples,
        num_refs,
        re_tessellation=100,
        permute_tests=100,
        z_score_norm=False,
        x_frac=None,
        kernel="euclidean",
        return_type="chi2",
    )
    print("pvalue:", np.mean(np.mean(res[1], axis=1) > np.mean(res[0])))
    print("Time taken (fast):", time() - start)
    start = time()
    res = permute_retesselate_pqm_test_slow(
        x_samples,
        y_samples,
        num_refs,
        re_tessellation=100,
        permute_tests=100,
        z_score_norm=False,
        x_frac=None,
        kernel="euclidean",
        return_type="chi2",
    )
    print("pvalue:", np.mean(np.mean(res[1], axis=1) > np.mean(res[0])))
    print("Time taken (slow):", time() - start)
