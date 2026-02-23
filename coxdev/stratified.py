import numpy as np
from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from scipy.sparse.linalg import LinearOperator
from joblib import hash as _hash

from .base import (CoxDevianceResult,
                   CoxInformation)
from .coxc import c_preprocess, cox_dev as _cox_dev, compute_sat_loglik as _compute_sat_loglik

@dataclass
class StratifiedCoxDeviance:

    event: InitVar[np.ndarray]
    status: InitVar[np.ndarray]
    strata: InitVar[Optional[np.ndarray]] = None
    start: InitVar[Optional[np.ndarray]] = None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'

    def __post_init__(self,
                      event,
                      status,
                      strata,
                      start=None,
                      tie_breaking='efron'):
        """
        Initialize the CoxDeviance object with survival data.
        
        Parameters
        ----------
        event : np.ndarray
            Event times for each observation.
        status : np.ndarray
            Event indicators (1 for event, 0 for censored).
        start : np.ndarray, optional
            Start times for left-truncated data.
        """
        event = np.asarray(event).astype(float)

        status = np.asarray(status)
        if not set(np.unique(status)).issubset(set([0,1])):
            raise ValueError('status must be binary')
        self._status = status.astype(np.int32)
        self._event = event
        nevent = event.shape[0]

        if start is None:
            start = -np.ones(nevent) * np.inf
            self._have_start_times = False
        else:
            start = np.asarray(start, float)
            self._have_start_times = True

        self._start = start
        if strata is None:
            strata = np.zeros(event.shape[0], np.int32)
        else:
            strata = np.asarray(strata)
        if not np.issubdtype(strata.dtype, np.integer):
            raise ValueError(f"strata must be integer type, got {strata.dtype}")
        self._strata = np.asarray(strata, np.int32)
        
        if ((status.shape != event.shape)
            or (strata.shape != event.shape)
            or (start.shape != event.shape)):
            raise ValueError("status, event, start and strata must have same shape")

        self._unique_strata = np.unique(strata)
        self._stratum_indices = {}
        for stratum in self._unique_strata:
            self._stratum_indices[stratum] = np.where(strata == stratum)[0]
        
        # Initialize result cache
        self._result = None
        self._last_hash = None

    def _setup_post_weights(self, sample_weight=None):

        event = self._event
        status = self._status
        start = self._start
        strata = self._strata
        
        if start is None:
            start = -np.ones(n) * np.inf
            have_start = False
        else:
            start = np.asarray(start)
            have_start = True

        self._have_start_times = have_start
        self._efron = self.tie_breaking == 'efron'
        self._efron_stratum = []
        self._unique_strata = np.unique(strata)
        self._stratum_indices = [np.where(strata == s)[0] for s in self._unique_strata]
        self._n_strata = len(self._unique_strata)

        # Preprocess and allocate buffers for each stratum

        self._preproc = []
        self._event_order = []
        self._start_order = []
        self._status_list = []
        self._event_list = []
        self._start_list = []
        self._sample_weight = []
        self._first = []
        self._last = []
        self._scaling = []
        self._event_map = []
        self._start_map = []
        self._first_start = []
        self._T_1_term = []
        self._T_2_term = []
        self._event_reorder_buffers = []
        self._forward_cumsum_buffers = []
        self._forward_scratch_buffer = []
        self._reverse_cumsum_buffers = []
        self._risk_sum_buffers = []
        self._hess_matvec_buffer = []
        self._grad_buffer = []
        self._diag_hessian_buffer = []
        self._diag_part_buffer = []
        self._w_avg_buffer = []
        self._exp_w_buffer = []

        # allocate and preprocess

        for idx in self._stratum_indices:
            e = event[idx]
            s = status[idx]
            st = start[idx]
            w = sample_weight[idx]
            preproc, event_order, start_order = c_preprocess(st, e, s, w)
            self._efron_stratum.append(self._efron and (np.linalg.norm(preproc['scaling']) > 0))
            n_stratum = len(idx)
            self._preproc.append(preproc)
            self._event_order.append(event_order.astype(np.int32))
            self._start_order.append(start_order.astype(np.int32))
            self._status_list.append(np.asarray(preproc['status']))
            self._event_list.append(np.asarray(preproc['event']))
            self._start_list.append(np.asarray(preproc['start']))
            self._first.append(np.asarray(preproc['first']).astype(np.int32))
            self._last.append(np.asarray(preproc['last']).astype(np.int32))
            self._scaling.append(np.asarray(preproc['scaling']))
            self._event_map.append(np.asarray(preproc['event_map']).astype(np.int32))
            self._start_map.append(np.asarray(preproc['start_map']).astype(np.int32))
            self._first_start.append(self._first[-1][self._start_map[-1]])
            self._T_1_term.append(np.zeros(n_stratum))
            self._T_2_term.append(np.zeros(n_stratum))
            self._event_reorder_buffers.append([np.zeros(n_stratum) for _ in range(3)])
            self._forward_cumsum_buffers.append([np.zeros(n_stratum+1) for _ in range(5)])
            self._forward_scratch_buffer.append(np.zeros(n_stratum))
            self._reverse_cumsum_buffers.append([np.zeros(n_stratum+1) for _ in range(4)])
            self._risk_sum_buffers.append([np.zeros(n_stratum) for _ in range(2)])
            self._hess_matvec_buffer.append(np.zeros(n_stratum))
            self._grad_buffer.append(np.zeros(n_stratum))
            self._diag_hessian_buffer.append(np.zeros(n_stratum))
            self._diag_part_buffer.append(np.zeros(n_stratum))
            self._w_avg_buffer.append(np.zeros(n_stratum))
            self._exp_w_buffer.append(np.zeros(n_stratum))

    """
    Stratified Cox Proportional Hazards Model Deviance Calculator.

    Efficiently computes deviance, gradient, and block-diagonal information matrix for the
    stratified Cox model, supporting Efron and Breslow tie-breaking and left truncation.

    Parameters
    ----------
    event : np.ndarray
        Event (failure) times.
    status : np.ndarray
        Event indicators (1=event, 0=censored).
    strata : np.ndarray, optional
        Stratum labels for each observation.
    start : np.ndarray, optional
        Start times for left-truncated data.
    tie_breaking : {'efron', 'breslow'}, default='efron'
        Tie-breaking method.

    Examples
    --------
    >>> import numpy as np
    >>> from coxdev import StratifiedCoxDeviance
    >>> event = np.array([3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4])
    >>> status = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    >>> strata = np.repeat([0, 1, 2], 4)
    >>> cox = StratifiedCoxDeviance(event=event, status=status, strata=strata)
    >>> eta = np.linspace(-1, 1, len(event))
    >>> result = cox(eta)
    >>> print(round(result.deviance, 4))
    14.2741
    """
    def __call__(self, linear_predictor, sample_weight=None):
        linear_predictor = np.asarray(linear_predictor)
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight)

        if not hasattr(self, "_count"):
            self._count = 0

        cur_hash = _hash(sample_weight)
        if hasattr(self, "_weights_hash"):
            if self._weights_hash != cur_hash:
                self._setup_post_weights(sample_weight)
                self._weights_hash = cur_hash
        else:
            self._setup_post_weights(sample_weight)
            self._weights_hash = cur_hash
            
        cur_hash = _hash([linear_predictor, sample_weight])

        if self._last_hash != cur_hash:
            # Prepare outputs
            deviance = 0.0
            loglik_sat = 0.0
            grad = np.zeros_like(linear_predictor)
            diag_hess = np.zeros_like(linear_predictor)
            # Loop over strata

            for i in range(len(self._stratum_indices)):
                idx = self._stratum_indices[i]
                eta = linear_predictor[idx]
                weight = sample_weight[idx]
                eta = eta - eta.mean()
                self._exp_w_buffer[i][:] = weight * np.exp(np.clip(eta, -np.inf, 30))
                loglik_sat_i = _compute_sat_loglik(
                    self._first[i], self._last[i], weight, self._event_order[i], self._status_list[i], self._forward_cumsum_buffers[i][0]
                )

                loglik_sat += loglik_sat_i
                dev = _cox_dev(
                    eta,
                    weight,
                    self._exp_w_buffer[i],
                    self._event_order[i],
                    self._start_order[i],
                    self._status_list[i],
                    self._first[i],
                    self._last[i],
                    self._scaling[i],
                    self._event_map[i],
                    self._start_map[i],
                    loglik_sat_i,
                    self._T_1_term[i],
                    self._T_2_term[i],
                    self._grad_buffer[i],
                    self._diag_hessian_buffer[i],
                    self._diag_part_buffer[i],
                    self._w_avg_buffer[i],
                    self._event_reorder_buffers[i],
                    self._risk_sum_buffers[i],
                    self._forward_cumsum_buffers[i],
                    self._forward_scratch_buffer[i],
                    self._reverse_cumsum_buffers[i],
                    self._have_start_times,
                    self._efron_stratum[i]
                )
                deviance += dev
                grad[idx] = self._grad_buffer[i]
                diag_hess[idx] = self._diag_hessian_buffer[i]

            self._result = CoxDevianceResult(
                linear_predictor=linear_predictor,
                sample_weight=sample_weight,
                loglik_sat=loglik_sat,
                deviance=deviance,
                gradient=grad,
                diag_hessian=diag_hess,
                __hash_args__=""
            )

            self._last_hash = cur_hash

        return self._result

    def information(self, linear_predictor, sample_weight=None):
        """Return a block-diagonal LinearOperator representing the information matrix."""
        return StratifiedCoxInformation(self, linear_predictor, sample_weight)


class StratifiedCoxInformation(LinearOperator):

    def __init__(self, strat_cox, linear_predictor, sample_weight):
        self.strat_cox = strat_cox
        self.linear_predictor = np.asarray(linear_predictor)
        self.sample_weight = np.ones_like(self.linear_predictor) if sample_weight is None else np.asarray(sample_weight)
        self.n = self.linear_predictor.shape[0]
        self.shape = (self.n, self.n)
        self.dtype = float
        # Precompute per-stratum information operators
        self._block_infos = []
        for i in range(len(self.strat_cox._stratum_indices)):
            # Use the same buffers as in __call__
            idx = self.strat_cox._stratum_indices[i]
            eta = self.linear_predictor[idx]
            weight = self.sample_weight[idx]
            # Call __call__ to ensure buffers are up to date
            self.strat_cox.__call__(self.linear_predictor, self.sample_weight)
            # Build a LinearOperator for this block
            # We mimic the logic from StratifiedCoxDeviance: use the same CoxInformation logic

            # Build a fake CoxDevianceResult for this block
            result = CoxDevianceResult(
                linear_predictor=eta,
                sample_weight=weight,
                loglik_sat=0.0,  # not used
                deviance=0.0,    # not used
                gradient=self.strat_cox._grad_buffer[i],
                diag_hessian=self.strat_cox._diag_hessian_buffer[i],
                __hash_args__=""
            )
            # Build a fake CoxDeviance-like object for this block
            class BlockCoxDev:
                pass
            blockdev = BlockCoxDev()
            blockdev._status = self.strat_cox._status_list[i]
            blockdev._risk_sum_buffers = self.strat_cox._risk_sum_buffers[i]
            blockdev._diag_part_buffer = self.strat_cox._diag_part_buffer[i]
            blockdev._w_avg_buffer = self.strat_cox._w_avg_buffer[i]
            blockdev._exp_w_buffer = self.strat_cox._exp_w_buffer[i]
            blockdev._event_cumsum = self.strat_cox._reverse_cumsum_buffers[i][0]
            blockdev._start_cumsum = self.strat_cox._reverse_cumsum_buffers[i][1]
            blockdev._event_order = self.strat_cox._event_order[i]
            blockdev._start_order = self.strat_cox._start_order[i]
            blockdev._first = self.strat_cox._first[i]
            blockdev._last = self.strat_cox._last[i]
            blockdev._scaling = self.strat_cox._scaling[i]
            blockdev._event_map = self.strat_cox._event_map[i]
            blockdev._start_map = self.strat_cox._start_map[i]
            blockdev._forward_cumsum_buffers = self.strat_cox._forward_cumsum_buffers[i]
            blockdev._forward_scratch_buffer = self.strat_cox._forward_scratch_buffer[i]
            blockdev._reverse_cumsum_buffers = self.strat_cox._reverse_cumsum_buffers[i]
            blockdev._hess_matvec_buffer = self.strat_cox._hess_matvec_buffer[i]
            blockdev._have_start_times = self.strat_cox._have_start_times
            blockdev._efron = self.strat_cox._efron
            # Use CoxInformation
            block_info = CoxInformation(result=result, coxdev=blockdev)
            self._block_infos.append((idx, block_info))

    def _matvec(self, v):
        v = np.asarray(v).reshape(-1)
        result = np.zeros_like(v)
        for idx, block_info in self._block_infos:
            result[idx] = block_info @ v[idx]
        return result

    def _adjoint(self, v):
        return self._matvec(v)

