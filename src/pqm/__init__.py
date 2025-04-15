from .pqm import pqm_pvalue, pqm_chi2
from .test_gaussian import test
from . import utils

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0-dev"


__all__ = (
    "pqm_pvalue",
    "pqm_chi2",
    "test",
    "utils",
    "__version__",
)
