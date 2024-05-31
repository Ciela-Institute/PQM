# PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

Implementation of the PQMass two sample test from Lemos et al. 2024 [here](https://arxiv.org/abs/2402.04355)

## Install

Just do:

```
pip install pqm
```

## Usage

This is the main use case:

```python
from pqm import pqm_pvalue
import numpy as np

x_sample = np.random.normal(size = (500, 10))
y_sample = np.random.normal(size = (400, 10))

# To get pvalues from PQMass
pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50)
print(np.mean(pvalues), np.std(pvalues))

# To get chi^2 from PQMass
chi2_stat, dof = pqm_chi2(x_sample, y_sample, num_refs = 100, bootstrap = 50)
print(np.mean(chi2_stat), np.std(chi2_stat))
print(np.unqiue(dof)) # This should be the same as num_refs - 1, if it is not, we suggest you use pqm_pvalue
```

If your two samples are drawn from the same distribution, then the p-value should
be drawn from the random uniform(0,1) distribution. This means that if you get a
very small value (i.e., 1e-6), then you have failed the null hypothesis test, and
the two samples are not drawn from the same distribution.

For the chi^2 metric, given your two sets of samples, if they come from the same
distribution, the histogram of your chi² values should follow the chi² distribution. 
The peak of this distribution will be at DoF - 2, and the standard deviation will 
be √(2 * DoF). If your histogram shifts to the right of the expected chi² distribution, 
it suggests that the samples are out of distribution. Conversely, if the histogram shifts 
to the left, it indicates potential duplication or memorization (particularly relevant 
for generative models).

Note that the chi^2 metric faces limitations if you have a few samples. A solution could
be to use bootstrapping. Another such solution is to pqm_pvalue. We leave it to the user to 
identify the best solution for their problem.

## Developing

If you're a developer then:

```
git clone git@github.com:Ciela-Institute/PQM.git
cd PQM
git checkout -b my-new-branch
pip install -e .
```
