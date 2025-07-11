"""
coxdev: Efficient Cox Proportional Hazards Model Deviance and Information

This package provides efficient computation of Cox model deviance, gradients, and information matrices
for survival analysis, including support for stratified models and different tie-breaking methods.

Main Classes
------------
CoxDeviance
    Standard Cox model deviance and information computation.
StratifiedCoxDeviance
    Stratified Cox model deviance and information computation.

See Also
--------
coxdev.base : Core Cox model implementation.
coxdev.stratified : Stratified Cox model implementation.
"""

from .base import CoxDeviance
from .stratified import StratifiedCoxDeviance
