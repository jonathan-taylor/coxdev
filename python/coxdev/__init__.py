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

from .base import CoxDeviance, BaselineHazard, SurvivalCurves


def get_include():
    """
    The directory of coxdev's C++ headers, for packages that compile against
    coxdev. coxdev is header-only: ``coxdev.h`` declares the classes
    ``CoxDeviance`` and ``StratifiedCoxDeviance`` and includes their
    implementation, all inline, from ``coxdev_impl.h``.

    Add this directory, and Eigen's headers, to the include path, and
    ``#include <coxdev.h>`` in any source files that use the classes. In a
    meson build::

        coxdev_include = run_command(py, '-c',
            'import coxdev; print(coxdev.get_include())', check : true).stdout().strip()

    with ``coxdev`` in ``[build-system] requires``.

    Returns
    -------
    str
        The path of the directory.
    """
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')
