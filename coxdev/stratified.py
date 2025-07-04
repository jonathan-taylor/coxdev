"""
Stratified Cox Proportional Hazards Model Deviance Computation.

This module provides stratified Cox model deviance computation, where
the baseline hazard is allowed to vary across strata while the regression
coefficients are shared across all strata.
"""

from dataclasses import dataclass, InitVar
from typing import Literal, Optional, Dict, List
import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import LinearOperator

from . import CoxDeviance, CoxDevianceResult, CoxInformation

from joblib import hash as _hash

@dataclass
class StratifiedCoxDeviance(object):
    """
    Stratified Cox Proportional Hazards Model Deviance Calculator.
    
    This class provides efficient computation of stratified Cox model deviance,
    gradients, and Hessian information matrices. It supports both Efron and Breslow
    tie-breaking methods and handles left-truncated survival data.
    
    In a stratified Cox model, the baseline hazard is allowed to vary across
    strata while the regression coefficients are shared across all strata.
    
    Parameters
    ----------
    event : np.ndarray
        Event times (failure times) for each observation.
    status : np.ndarray
        Event indicators (1 for event occurred, 0 for censored).
    strata : np.ndarray
        Stratum indicators for each observation. Must be np.int32.
    start : np.ndarray, optional
        Start times for left-truncated data. If None, assumes no truncation.
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Method for handling tied event times.
        
    Attributes
    ----------
    tie_breaking : str
        The tie-breaking method being used.
    _have_start_times : bool
        Whether start times are provided.
    _efron : bool
        Whether Efron's method is being used for tie-breaking.
    _strata : np.ndarray
        The stratum indicators.
    _unique_strata : np.ndarray
        Unique stratum values.
    _stratum_indices : Dict[int, np.ndarray]
        Mapping from stratum value to indices of observations in that stratum.
    _stratum_coxdevs : Dict[int, CoxDeviance]
        Mapping from stratum value to CoxDeviance instance for that stratum.
    """
    
    event: InitVar[np.ndarray]
    status: InitVar[np.ndarray]
    strata: InitVar[np.ndarray] = None
    start: InitVar[np.ndarray] = None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'
    
    def __post_init__(self,
                      event,
                      status,
                      strata,
                      start=None):
        """
        Initialize the StratifiedCoxDeviance object with survival data.
        
        Parameters
        ----------
        event : np.ndarray
            Event times for each observation.
        status : np.ndarray
            Event indicators (1 for event, 0 for censored).
        strata : np.ndarray
            Stratum indicators for each observation.
        start : np.ndarray, optional
            Start times for left-truncated data.
        """
        # Convert status to int32 and validate
        status = np.asarray(status).astype(np.int32)
        if status.dtype != np.int32:
            raise ValueError(f"status must be convertible to int32, got {status.dtype}")
        
        # Convert strata to int32 and validate
        strata = np.asarray(strata).astype(np.int32)
        
        # Validate that strata is int32 after casting
        if strata.dtype != np.int32:
            raise ValueError(f"strata must be convertible to int32, got {strata.dtype}")
        
        # Validate input lengths
        if len(event) != len(status) or len(event) != len(strata):
            raise ValueError("event, status, and strata must have the same length")

        # Store the original data
        self._event = np.asarray(event)
        self._status = np.asarray(status)
        self._start = start
        
        # Get unique strata
        self._unique_strata = np.unique(strata)
        
        # Create mapping from stratum to indices
        self._stratum_indices = {}
        for stratum in self._unique_strata:
            self._stratum_indices[stratum] = np.where(strata == stratum)[0]
        
        # Create separate CoxDeviance instances for each stratum
        self._stratum_coxdevs = {}

        for stratum in self._unique_strata:
            indices = self._stratum_indices[stratum]
            
            # Extract data for this stratum
            stratum_event = self._event[indices]
            stratum_status = self._status[indices]
            stratum_start = None if start is None else start[indices]
            
            # Create CoxDeviance instance for this stratum
            self._stratum_coxdevs[stratum] = CoxDeviance(
                event=stratum_event,
                status=stratum_status,
                start=stratum_start,
                tie_breaking=self.tie_breaking
            )
        
        # Set up attributes for compatibility with parent class
        self._have_start_times = start is not None
        self._efron = self.tie_breaking == 'efron'
        
        # Initialize result cache
        self._result = None
        self._last_hash = None

    def __call__(self,
                 linear_predictor,
                 sample_weight=None):
        """
        Compute stratified Cox model deviance and related quantities.
        
        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor values (X @ beta).
        sample_weight : np.ndarray, optional
            Sample weights. If None, uses equal weights.
            
        Returns
        -------
        CoxDevianceResult
            Object containing deviance, gradient, and Hessian diagonal.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight)

        linear_predictor = np.asarray(linear_predictor)
        
        # Check if we need to recompute
        cur_hash = _hash([linear_predictor, sample_weight])
        if self._last_hash != cur_hash:
            
            # Initialize accumulators
            total_deviance = 0.0
            total_loglik_sat = 0.0
            total_gradient = np.zeros_like(linear_predictor)
            total_diag_hessian = np.zeros_like(linear_predictor)
            
            # Compute results for each stratum
            for stratum in self._unique_strata:
                indices = self._stratum_indices[stratum]
                coxdev = self._stratum_coxdevs[stratum]
                
                # Extract data for this stratum
                stratum_linear_predictor = linear_predictor[indices]
                stratum_sample_weight = sample_weight[indices]
                
                # Compute results for this stratum
                stratum_result = coxdev(stratum_linear_predictor, stratum_sample_weight)
                
                # Accumulate results
                total_deviance += stratum_result.deviance
                total_loglik_sat += stratum_result.loglik_sat
                total_gradient[indices] = stratum_result.gradient
                total_diag_hessian[indices] = stratum_result.diag_hessian
            
            # Create combined result
            self._result = CoxDevianceResult(
                linear_predictor=linear_predictor,
                sample_weight=sample_weight,
                loglik_sat=total_loglik_sat,
                deviance=total_deviance,
                gradient=total_gradient,
                diag_hessian=total_diag_hessian,
                __hash_args__=cur_hash
            )
            
            self._last_hash = cur_hash
        
        return self._result

    def information(self,
                    linear_predictor,
                    sample_weight=None):
        """
        Compute the information matrix (negative Hessian) as a linear operator.
        
        The information matrix is block diagonal by stratum, with each block
        given by the individual CoxDeviance.information blocks.
        
        Parameters
        ----------
        linear_predictor : np.ndarray
            Linear predictor values (X @ beta).
        sample_weight : np.ndarray, optional
            Sample weights. If None, uses equal weights.
            
        Returns
        -------
        StratifiedCoxInformation
            Linear operator representing the block diagonal information matrix.
        """
        result = self(linear_predictor, sample_weight)
        return StratifiedCoxInformation(result=result, stratified_coxdev=self)


@dataclass
class StratifiedCoxInformation(LinearOperator):
    """
    Linear operator representing the stratified Cox model information matrix.
    
    This class provides matrix-vector multiplication with the block diagonal
    information matrix (negative Hessian) of the stratified Cox model.
    
    Parameters
    ----------
    stratified_coxdev : StratifiedCoxDeviance
        The StratifiedCoxDeviance object used for computations.
    result : CoxDevianceResult
        Result from the most recent deviance computation.
        
    Attributes
    ----------
    shape : tuple
        Shape of the information matrix (n, n).
    dtype : type
        Data type of the matrix elements.
    """

    stratified_coxdev: StratifiedCoxDeviance
    result: CoxDevianceResult

    def __post_init__(self):
        """Initialize the linear operator dimensions."""
        n = len(self.result.linear_predictor)
        self.shape = (n, n)
        self.dtype = float
        
    def _matvec(self, arg):
        """
        Compute matrix-vector product with the block diagonal information matrix.
        
        Parameters
        ----------
        arg : np.ndarray
            Vector to multiply with the information matrix.
            
        Returns
        -------
        np.ndarray
            Result of the matrix-vector multiplication.
        """
        arg = np.asarray(arg).reshape(-1)
        result = np.zeros_like(arg)
        
        # Apply each stratum's information matrix to its corresponding indices
        for stratum in self.stratified_coxdev._unique_strata:
            indices = self.stratified_coxdev._stratum_indices[stratum]
            coxdev = self.stratified_coxdev._stratum_coxdevs[stratum]
            
            # Extract the part of arg corresponding to this stratum
            stratum_arg = arg[indices]
            
            # Get the information matrix for this stratum
            stratum_info = coxdev.information(
                self.result.linear_predictor[indices],
                self.result.sample_weight[indices]
            )
            
            # Apply the information matrix
            stratum_result = stratum_info @ stratum_arg
            
            # Store the result in the appropriate positions
            result[indices] = stratum_result
        
        return result

    def _adjoint(self, arg):
        """
        Compute the adjoint (transpose) matrix-vector product.
        
        Since the information matrix is symmetric, this is the same as _matvec.
        
        Parameters
        ----------
        arg : np.ndarray
            Vector to multiply with the adjoint matrix.
            
        Returns
        -------
        np.ndarray
            Result of the adjoint matrix-vector multiplication.
        """
        # The information matrix is symmetric
        return self._matvec(arg) 
