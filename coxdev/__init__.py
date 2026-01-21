"""
coxdev: Efficient Cox Proportional Hazards Model Deviance and Information

This package provides efficient computation of Cox model deviance, gradients, and information matrices
for survival analysis, including support for stratified models and different tie-breaking methods.

Main Classes
------------
CoxDeviance
    Standard Cox model deviance and information computation.
StratifiedCoxDeviance
    Stratified Cox model (wrapper around StratifiedCoxDevianceCpp for backward compatibility).
StratifiedCoxDevianceCpp
    Stratified Cox model with pure C++ strata processing.

See Also
--------
coxdev.base : Core Cox model implementation.
coxdev.stratified : Stratified Cox model implementation.
coxdev.stratified_cpp : C++ stratified Cox model implementation.
"""

from .base import CoxDeviance
from .stratified import StratifiedCoxDeviance
from .stratified_cpp import StratifiedCoxDevianceCpp
