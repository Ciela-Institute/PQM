# PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

![PyPI - Version](https://img.shields.io/pypi/v/pqm?style=flat-square)
[![CI](https://github.com/Ciela-Institute/PQM/actions/workflows/ci.yml/badge.svg)](https://github.com/Ciela-Institute/PQM/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pqm)
[![arXiv](https://img.shields.io/badge/arXiv-2402.04355-b31b1b.svg)](https://arxiv.org/abs/2402.04355)

Implementation of the PQMass two sample test from Lemos et al. 2024 [here](https://arxiv.org/abs/2402.04355)

## Install

Just do:

```
pip install pqm
```

## Usage

This is the main use case:

```python
from pqm import pqm_pvalue, pqm_chi2
import numpy as np

x_sample = np.random.normal(size = (500, 10))
y_sample = np.random.normal(size = (400, 10))

# To get pvalues from PQMass
pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, re_tessellation = 50)
print(np.mean(pvalues), np.std(pvalues))

# To get chi^2 from PQMass
chi2_stat = pqm_chi2(x_sample, y_sample, num_refs = 100, re_tessellation = 50)
print(np.mean(chi2_stat), np.std(chi2_stat))
```

If your two samples are drawn from the same distribution, then the p-value
should be drawn from the random uniform(0,1) distribution. This means that if
you get a very small value (i.e., 1e-6), then you have failed the null
hypothesis test, and the two samples are not drawn from the same distribution.
If you get values approximately equal to 1 every time then that suggests
potential duplication of samples between `x_samples` and `y_samples`.

For the chi^2 metric, given your two sets of samples, if they come from the same
distribution, the histogram of your chi^2 values should follow the chi^2
distribution. The degrees of freedom (DoF) will equal `DoF = num_refs - 1` The
peak of this distribution will be at `DoF - 2`, the mean will equal `DoF`, and
the standard deviation will be `sqrt(2 * DoF)`. If your chi^2 values are too
high (`chi^2 / DoF > 1`), it suggests that the samples are out of distribution.
Conversely, if the values are too low (`chi^2 / DoF < 1`), it indicates
potential duplication of samples between `x_samples` and `y_samples` (i.e.
memorization for generative models).


## Developing

If you're a developer then:

```
git clone git@github.com:Ciela-Institute/PQM.git
cd PQM
git checkout -b my-new-branch
pip install -e .
```

But make an issue first so we can discuss implementation ideas.