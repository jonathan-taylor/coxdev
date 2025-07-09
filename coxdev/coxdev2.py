import numpy as np
from dataclasses import dataclass, InitVar
from typing import Optional, Literal
from coxdev import CoxDevianceResult
from coxc import c_preprocess, cox_dev as _cox_dev, compute_sat_loglik as _compute_sat_loglik
from scipy.sparse.linalg import LinearOperator
from coxdev import CoxInformation, CoxDevianceResult

@dataclass
class CoxDeviance2:
    event: InitVar[np.ndarray]
    status: InitVar[np.ndarray]
    strata: InitVar[Optional[np.ndarray]] = None
    start: InitVar[Optional[np.ndarray]] = None
    tie_breaking: Literal['efron', 'breslow'] = 'efron'

    def __post_init__(self, event, status, strata=None, start=None):
        event = np.asarray(event)
        status = np.asarray(status).astype(np.int32)
        n = event.shape[0]

        if strata is None:
            strata = np.zeros(n, dtype=np.int32)
        else:
            strata = np.asarray(strata).astype(np.int32)
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

        # Store for later
        self._strata = strata
        self._event = event
        self._status = status
        self._start = start

        # Preprocess and allocate buffers for each stratum

        self._preproc = []
        self._event_order = []
        self._start_order = []
        self._status_list = []
        self._event_list = []
        self._start_list = []
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
            preproc, event_order, start_order = c_preprocess(st, e, s)
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

    def __call__(self, linear_predictor, sample_weight=None):
        linear_predictor = np.asarray(linear_predictor)
        if sample_weight is None:
            sample_weight = np.ones_like(linear_predictor)
        else:
            sample_weight = np.asarray(sample_weight)
        # Prepare outputs
        deviance = 0.0
        loglik_sat = 0.0
        grad = np.zeros_like(linear_predictor)
        diag_hess = np.zeros_like(linear_predictor)
        # Loop over strata
        for i, idx in enumerate(self._stratum_indices):
            eta = linear_predictor[idx]
            weight = sample_weight[idx]
            eta = eta - eta.mean()
            self._exp_w_buffer[i][:] = weight * np.exp(np.clip(eta, -np.inf, 30))
            loglik_sat_i = _compute_sat_loglik(
                self._first[i], self._last[i], weight, self._event_order[i], self._status_list[i], self._forward_cumsum_buffers[i][0]
            )
            # Print arguments to _cox_dev for debugging
            stratum_id = self._unique_strata[i]
            arg_dict = {
                'eta': eta,
                'sample_weight': weight,
                'exp_w': self._exp_w_buffer[i],
                'event_order': self._event_order[i],
                'start_order': self._start_order[i],
                'status': self._status_list[i],
                'first': self._first[i],
                'last': self._last[i],
                'scaling': self._scaling[i],
                'event_map': self._event_map[i],
                'start_map': self._start_map[i],
                'loglik_sat': loglik_sat_i,
                'T_1_term': self._T_1_term[i],
                'T_2_term': self._T_2_term[i],
                'grad_buffer': self._grad_buffer[i],
                'diag_hessian_buffer': self._diag_hessian_buffer[i],
                'diag_part_buffer': self._diag_part_buffer[i],
                'w_avg_buffer': self._w_avg_buffer[i],
                'event_reorder_buffers': self._event_reorder_buffers[i],
                'risk_sum_buffers': self._risk_sum_buffers[i],
                'forward_cumsum_buffers': self._forward_cumsum_buffers[i],
                'forward_scratch_buffer': self._forward_scratch_buffer[i],
                'reverse_cumsum_buffers': self._reverse_cumsum_buffers[i],
                'have_start_times': self._have_start_times,
                'efron': self._efron_stratum[i]
            }
            # Create debug dataframe for this stratum
            debug_data = {}
            
            # Find the maximum length among all arrays
            max_length = 0
            for name, arr in arg_dict.items():
                if isinstance(arr, (np.ndarray, list)):
                    if isinstance(arr, list):
                        # For lists, find the maximum length across all entries
                        for item in arr:
                            if isinstance(item, np.ndarray):
                                max_length = max(max_length, len(item))
                    else:
                        max_length = max(max_length, len(arr))
            
            for name, arr in arg_dict.items():
                if isinstance(arr, (np.ndarray, list)):
                    if isinstance(arr, list):
                        # For lists, include all entries
                        for j, item in enumerate(arr):
                            if isinstance(item, np.ndarray):
                                # Pad with zeros to match max_length
                                if len(item) < max_length:
                                    padded_item = np.pad(item, (0, max_length - len(item)), 'constant')
                                else:
                                    padded_item = item
                                debug_data[f"{name}_{j}"] = padded_item
                    else:
                        # Pad with zeros to match max_length
                        if len(arr) < max_length:
                            padded_arr = np.pad(arr, (0, max_length - len(arr)), 'constant')
                        else:
                            padded_arr = arr
                        debug_data[name] = padded_arr
                else:
                    # For scalar values, repeat for the max_length
                    debug_data[name] = [arr] * max_length
            
            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(debug_data)
            df.to_csv(f"CoxDev2{stratum_id}.csv", index=False)
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

        return CoxDevianceResult(
            linear_predictor=linear_predictor,
            sample_weight=sample_weight,
            loglik_sat=loglik_sat,
            deviance=deviance,
            gradient=grad,
            diag_hessian=diag_hess,
            __hash_args__=""
        )


    def information(self, linear_predictor, sample_weight=None):
        """Return a block-diagonal LinearOperator representing the information matrix."""
        return CoxDeviance2Information(self, linear_predictor, sample_weight)


class CoxDeviance2Information(LinearOperator):

    def __init__(self, coxdev2, linear_predictor, sample_weight):
        self.coxdev2 = coxdev2
        self.linear_predictor = np.asarray(linear_predictor)
        self.sample_weight = np.ones_like(self.linear_predictor) if sample_weight is None else np.asarray(sample_weight)
        self.n = self.linear_predictor.shape[0]
        self.shape = (self.n, self.n)
        self.dtype = float
        # Precompute per-stratum information operators
        self._block_infos = []
        for i, idx in enumerate(self.coxdev2._stratum_indices):
            # Use the same buffers as in __call__
            eta = self.linear_predictor[idx]
            weight = self.sample_weight[idx]
            # Call __call__ to ensure buffers are up to date
            self.coxdev2.__call__(self.linear_predictor, self.sample_weight)
            # Build a LinearOperator for this block
            # We mimic the logic from StratifiedCoxDeviance: use the same CoxInformation logic

            # Build a fake CoxDevianceResult for this block
            result = CoxDevianceResult(
                linear_predictor=eta,
                sample_weight=weight,
                loglik_sat=0.0,  # not used
                deviance=0.0,    # not used
                gradient=self.coxdev2._grad_buffer[i],
                diag_hessian=self.coxdev2._diag_hessian_buffer[i],
                __hash_args__=""
            )
            # Build a fake CoxDeviance-like object for this block
            class BlockCoxDev:
                pass
            blockdev = BlockCoxDev()
            blockdev._status = self.coxdev2._status_list[i]
            blockdev._risk_sum_buffers = self.coxdev2._risk_sum_buffers[i]
            blockdev._diag_part_buffer = self.coxdev2._diag_part_buffer[i]
            blockdev._w_avg_buffer = self.coxdev2._w_avg_buffer[i]
            blockdev._exp_w_buffer = self.coxdev2._exp_w_buffer[i]
            blockdev._event_cumsum = self.coxdev2._reverse_cumsum_buffers[i][0]
            blockdev._start_cumsum = self.coxdev2._reverse_cumsum_buffers[i][1]
            blockdev._event_order = self.coxdev2._event_order[i]
            blockdev._start_order = self.coxdev2._start_order[i]
            blockdev._first = self.coxdev2._first[i]
            blockdev._last = self.coxdev2._last[i]
            blockdev._scaling = self.coxdev2._scaling[i]
            blockdev._event_map = self.coxdev2._event_map[i]
            blockdev._start_map = self.coxdev2._start_map[i]
            blockdev._forward_cumsum_buffers = self.coxdev2._forward_cumsum_buffers[i]
            blockdev._forward_scratch_buffer = self.coxdev2._forward_scratch_buffer[i]
            blockdev._reverse_cumsum_buffers = self.coxdev2._reverse_cumsum_buffers[i]
            blockdev._hess_matvec_buffer = self.coxdev2._hess_matvec_buffer[i]
            blockdev._have_start_times = self.coxdev2._have_start_times
            blockdev._efron = self.coxdev2._efron
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

