# PQM

Implementation of the PQMass two sample test from Lemos et al. 2024

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

pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50)

print(np.mean(pvalues), np.std(pvalues))
```

If your two samples are drawn from the same distribution then the pvalue should
be drawn from the random uniform(0,1) distribution. This means that if you get a
very small value (i.e. 1e-6) then you have failed the null hypothesis test and
the two samples are not drawn form the same distribution.

## Developing

If you're a developer then:

```
git clone git@github.com:Ciela-Institute/PQM.git
cd PQM
git checkout -b my-new-branch
pip install -e .
```
