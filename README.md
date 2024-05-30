# PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

Implementation of the PQMass two sample test from Lemos et al. 2024 [here](https://arxiv.org/abs/2402.04355)

## Install

Just do:

```
pip install pqm
```

## Usage

```python
from pqm import pqm_pvalue
import numpy as np

x_sample = np.random.normal(size = (500, 10))
y_sample = np.random.normal(size = (400, 10))

# To get pvalues from PQMass
pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50)
print(np.mean(pvalues), np.std(pvalues))

# To get chi2 from PQMass
chi2_stat = pqm_chi2(x_sample, y_sample, num_refs = 100, bootstrap = 50)
print(np.mean(chi2_stat), np.std(chi2_stat))
```

If your two samples are drawn from the same distribution then the pvalue should
be drawn from the random uniform(0,1) distribution. This means that if you get a
very small value (i.e. 1e-6) then you have failed the null hypothesis test and
the two samples are not drawn from the same distribution.

For the chi2 metric, given your two sets of samples, if they come from the same
distribution, the histogram of your chi² values should follow the chi² distribution. 
The peak of this distribution will be at DoF -  and the standard deviation will 
be √(2 * DoF). If your histogram shifts to the right of the expected chi² distribution, 
it suggests that the samples are out of distribution. Conversely, if the histogram shifts 
to the left, it indicates potential duplication or memorization (particularly relevant 
for generative models).

Note that if you have low number of samples, we highly recommend using pvalue as the 
chi2 metric can be biased in the low number of sample regime.

## Developing

If you're a developer then:

```
git clone git@github.com:Ciela-Institute/PQM.git
cd PQM
git checkout -b my-new-branch
pip install -e .
```