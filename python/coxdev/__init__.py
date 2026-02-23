"""
coxdev: Efficient Cox Proportional Hazards Model Deviance and Information

This package provides efficient computation of Cox model deviance, gradients, and information matrices
for survival analysis, including support for stratified models and different tie-breaking methods.

Main Classes
------------
CoxDeviance
    Standard Cox model deviance and information computation.

See Also
--------
coxdev.base : Core Cox model implementation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("coxdev")
except PackageNotFoundError:
    # package is not installed, perhaps we are in a git repo
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "unknown"

from .base import CoxDeviance

