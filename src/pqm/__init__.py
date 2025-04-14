from .pqm import pqm
from .test_gaussian import test

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0-dev"


__all__ = (
    "pqm",
    "test",
    "__version__",
)
