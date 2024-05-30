# PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

Implementation of the PQMass two sample test from Lemos et al. 2024 [here](https://arxiv.org/abs/2402.04355)

## Usage

```python
from pqm import pqm_pvalue
import numpy as np

x_sample = np.random.normal(size = (500, 10))
y_sample = np.random.normal(size = (400, 10))

# To get pvalues from PQMass (this is the default case so you do not need to explicity write return_stat = "pvalue")
pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50, return_stat = "pvalue")
print(np.mean(pvalues), np.std(pvalues))

# To get chi2 from PQMass
chi2_stat = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50, return_stat = "chi2")
print(np.mean(chi2_stat), np.std(chi2_stat))

# Can get both chi2 and pvalue 
chi2_stat, pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50, return_stat = "both")
print(np.mean(pvalues), np.std(pvalues))
print(np.mean(chi2_stat), np.std(chi2_stat))
```