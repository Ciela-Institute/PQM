# PQM

Implementation of the PQMass two sample test from Lemos et al. 2024

## Usage

```python
from pqm import pqm_pvalue
import numpy as np

x_sample = np.random.normal(size = (500, 10))
y_sample = np.random.normal(size = (400, 10))

pvalues = pqm_pvalue(x_sample, y_sample, num_refs = 100, bootstrap = 50)

print(np.mean(pvalues), np.std(pvalues))
```