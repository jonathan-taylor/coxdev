"""
Progressive tolerance tests: coxdev vs R adelie.

Instead of a single pass/fail threshold, computes raw numerical discrepancies
between coxdev and R adelie across a grid of scenarios and reports which
tolerance levels pass/fail. Hard-asserts at 1e-10 (regression guard);
reports-only for tighter levels.

Run:
    uv run pytest tests/test_tolerance_progression.py -v -s
    uv run pytest tests/test_tolerance_progression.py::test_summary_table -v -s
    uv run pytest tests/test_tolerance_progression.py::TestTightTolerances -v -s
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from coxdev import CoxDeviance, StratifiedCoxDeviance

# ---------------------------------------------------------------------------
# R adelie via rpy2 (self-contained copy to avoid pytest collection issues)
# ---------------------------------------------------------------------------
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    adelie_r = importr("adelie")
    HAS_ADELIE = True
except (ImportError, Exception):
    HAS_ADELIE = False

pytestmark = pytest.mark.skipif(
    not HAS_ADELIE, reason="R adelie (via rpy2) not available"
)


def r_adelie_cox(start, stop, status, weights, tie_method, strata=None):
    """Create R adelie Cox GLM object and return a Python wrapper."""
    ro.globalenv["rstart"] = ro.FloatVector(start)
    ro.globalenv["rstop"] = ro.FloatVector(stop)
    ro.globalenv["rstatus"] = ro.FloatVector(status)
    ro.globalenv["rweights"] = ro.FloatVector(weights)
    ro.globalenv["rtie"] = tie_method

    if strata is not None:
        ro.globalenv["rstrata"] = ro.IntVector(strata + 1)
        ro.r(
            "rglm <- glm.cox(stop=rstop, status=rstatus, start=rstart, "
            "weights=rweights, tie_method=rtie, strata=rstrata)"
        )
    else:
        ro.r(
            "rglm <- glm.cox(stop=rstop, status=rstatus, start=rstart, "
            "weights=rweights, tie_method=rtie)"
        )
    return _RadelieCox()


class _RadelieCox:
    """Thin wrapper around the R adelie GLM object stored in R globalenv."""

    def loss(self, eta):
        ro.globalenv["reta"] = ro.FloatVector(eta)
        return float(ro.r("rglm$loss(reta)")[0])

    def loss_full(self):
        return float(ro.r("rglm$loss_full()")[0])

    def gradient(self, eta):
        ro.globalenv["reta"] = ro.FloatVector(eta)
        return np.array(ro.r("rglm$gradient(reta)"))

    def hessian(self, eta, grad):
        ro.globalenv["reta"] = ro.FloatVector(eta)
        ro.globalenv["rgrad"] = ro.FloatVector(grad)
        return np.array(ro.r("rglm$hessian(reta, rgrad)"))


# ---------------------------------------------------------------------------
# Tolerance grid
# ---------------------------------------------------------------------------
TOLERANCE_GRID = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
REGRESSION_TOL = 1e-10  # hard-assert level


# ---------------------------------------------------------------------------
# Dataclasses for structured metrics
# ---------------------------------------------------------------------------
@dataclass
class DiscrepancyMetrics:
    """Element-wise discrepancy between two vectors (or scalars)."""

    max_abs: float = 0.0
    max_rel: float = 0.0
    L2_rel: float = 0.0

    def passes(self, tol: float) -> bool:
        return self.max_rel <= tol or self.max_abs <= tol


@dataclass
class ScenarioReport:
    """Full discrepancy report for one scenario."""

    label: str
    gradient: DiscrepancyMetrics = field(default_factory=DiscrepancyMetrics)
    hessian: DiscrepancyMetrics = field(default_factory=DiscrepancyMetrics)
    loglik_sat_abs: float = 0.0
    loglik_sat_rel: float = 0.0

    def tolerance_pass(self, tol: float) -> bool:
        # For loglik_sat: pass if abs diff is small OR rel diff is small.
        # When both values are near zero (e.g. no-ties case where sat loglik
        # is exactly 0), relative error is meaningless.
        sat_ok = self.loglik_sat_abs <= tol or self.loglik_sat_rel <= tol
        return (
            self.gradient.passes(tol)
            and self.hessian.passes(tol)
            and sat_ok
        )

    def summary_line(self) -> str:
        cols = []
        for tol in TOLERANCE_GRID:
            cols.append("pass" if self.tolerance_pass(tol) else "FAIL")
        return f"  {self.label:<60s}  " + "  ".join(f"{c:>5s}" for c in cols)


# Module-level collector for the summary table
_all_reports: list[ScenarioReport] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_vector_discrepancy(a: np.ndarray, b: np.ndarray) -> DiscrepancyMetrics:
    """Compute max-abs, max-rel, L2-rel discrepancy between vectors a and b."""
    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size > 0 else 0.0

    denom = np.maximum(np.abs(a), np.abs(b))
    safe = denom > 0
    if np.any(safe):
        max_rel = float(np.max(diff[safe] / denom[safe]))
    else:
        max_rel = 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_max = max(norm_a, norm_b)
    L2_rel = float(np.linalg.norm(a - b) / norm_max) if norm_max > 0 else 0.0

    return DiscrepancyMetrics(max_abs=max_abs, max_rel=max_rel, L2_rel=L2_rel)


def compute_scalar_discrepancy(a: float, b: float):
    """Return (abs_diff, rel_diff) for two scalars."""
    abs_diff = abs(a - b)
    denom = max(abs(a), abs(b))
    rel_diff = abs_diff / denom if denom > 0 else 0.0
    return abs_diff, rel_diff


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
def generate_scenario_data(
    n: int,
    tie_pattern: str,  # "none", "moderate", "heavy"
    weight_type: str,  # "uniform", "random", "random_zeros"
    left_truncation: bool,
    stratified: bool,
    seed: int,
):
    """
    Generate synthetic survival data for a given scenario.

    Returns dict with keys: start, stop, status, eta, weights, strata (or None).
    """
    rng = np.random.default_rng(seed)

    # --- Event times ---
    if tie_pattern == "none":
        stop = np.sort(rng.exponential(2, n)) + 0.01 * np.arange(n)
    elif tie_pattern == "moderate":
        n_unique = max(2, int(n * 0.7))
        unique_times = np.sort(rng.exponential(2, n_unique))
        stop = rng.choice(unique_times, n)
    else:  # heavy
        n_unique = max(2, int(n * 0.2))
        unique_times = np.sort(rng.exponential(2, n_unique))
        stop = rng.choice(unique_times, n)

    # --- Start times (left truncation) ---
    if left_truncation:
        start = rng.exponential(0.5, n)
        stop = stop + start  # ensure stop > start
    else:
        start = np.zeros(n)

    # Ensure strict positivity
    stop = np.maximum(stop, start + 0.01)

    # --- Status ---
    status = rng.binomial(1, 0.6, n).astype(float)
    # Ensure at least one event
    if status.sum() == 0:
        status[rng.integers(n)] = 1.0

    # --- Weights ---
    if weight_type == "uniform":
        weights = np.ones(n)
    elif weight_type == "random":
        weights = rng.exponential(1, n)
    else:  # random_zeros
        weights = rng.exponential(1, n)
        n_zero = max(1, int(0.2 * n))
        zero_idx = rng.choice(n, size=n_zero, replace=False)
        weights[zero_idx] = 0.0

    # --- eta ---
    eta = rng.normal(0, 0.5, n)

    # --- Strata ---
    strata = None
    if stratified:
        strata = rng.integers(0, 3, n)

    return {
        "start": start.astype(np.float64),
        "stop": stop.astype(np.float64),
        "status": status.astype(np.float64),
        "eta": eta.astype(np.float64),
        "weights": weights.astype(np.float64),
        "strata": strata,
    }


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------
def run_scenario(
    n: int,
    tie_method: str,
    tie_pattern: str,
    weight_type: str,
    left_truncation: bool,
    stratified: bool,
    seed: int,
) -> ScenarioReport:
    """Run coxdev + R adelie, convert conventions, return ScenarioReport."""

    strat_tag = "strat" if stratified else "unstrat"
    trunc_tag = "trunc" if left_truncation else "notrunc"
    label = (
        f"n={n}, {tie_method}, ties={tie_pattern}, {strat_tag}, "
        f"wt={weight_type}, {trunc_tag}, seed={seed}"
    )

    data = generate_scenario_data(
        n=n,
        tie_pattern=tie_pattern,
        weight_type=weight_type,
        left_truncation=left_truncation,
        stratified=stratified,
        seed=seed,
    )

    start = data["start"]
    stop = data["stop"]
    status = data["status"]
    eta = data["eta"]
    weights = data["weights"]
    strata = data["strata"]

    weight_sum = np.sum(weights)
    weighted_events = np.sum(weights * status)

    # --- coxdev ---
    if strata is not None:
        cox = StratifiedCoxDeviance(
            event=stop,
            status=status,
            strata=strata,
            start=start,
            sample_weight=weights,
            tie_breaking=tie_method,
        )
    else:
        cox = CoxDeviance(
            event=stop,
            status=status,
            start=start,
            sample_weight=weights,
            tie_breaking=tie_method,
        )

    res = cox(eta)
    grad_coxdev = res.gradient
    hess_coxdev = res.diag_hessian
    loglik_sat_coxdev = res.loglik_sat

    # --- R adelie ---
    adl = r_adelie_cox(
        start=start,
        stop=stop,
        status=status,
        weights=weights,
        tie_method=tie_method,
        strata=strata,
    )

    grad_adelie = adl.gradient(eta)
    hess_adelie = adl.hessian(eta, grad_adelie)
    loss_full_adelie = adl.loss_full()

    # --- Convention conversion ---
    grad_from_adelie = -2 * weight_sum * grad_adelie
    hess_from_adelie = 2 * weight_sum * hess_adelie

    if weight_sum > 0:
        loglik_sat_from_adelie = (
            -weight_sum * loss_full_adelie
            - np.log(weight_sum) * weighted_events
        )
    else:
        loglik_sat_from_adelie = 0.0

    # --- Discrepancies ---
    report = ScenarioReport(label=label)
    report.gradient = compute_vector_discrepancy(grad_coxdev, grad_from_adelie)
    report.hessian = compute_vector_discrepancy(hess_coxdev, hess_from_adelie)
    report.loglik_sat_abs, report.loglik_sat_rel = compute_scalar_discrepancy(
        loglik_sat_coxdev, loglik_sat_from_adelie
    )

    return report


# ---------------------------------------------------------------------------
# Parametrised scenario grid
# ---------------------------------------------------------------------------
_N_VALUES = [20, 100, 500]
_TIE_METHODS = ["efron", "breslow"]
_TIE_PATTERNS = ["none", "moderate", "heavy"]
_STRATA = [False, True]
_WEIGHT_TYPES = ["uniform", "random", "random_zeros"]
_TRUNCATION = [False, True]


def _build_scenario_params():
    """Build list of (params_dict, id_string) for pytest parametrize."""
    params = []
    for n in _N_VALUES:
        for tie_method in _TIE_METHODS:
            for tie_pattern in _TIE_PATTERNS:
                for stratified in _STRATA:
                    for weight_type in _WEIGHT_TYPES:
                        for left_truncation in _TRUNCATION:
                            # Key scenarios (n=100, moderate) get 3 seeds
                            if n == 100 and tie_pattern == "moderate":
                                seeds = [42, 123, 7]
                            else:
                                seeds = [42]
                            for seed in seeds:
                                strat_tag = "strat" if stratified else "unstrat"
                                trunc_tag = "trunc" if left_truncation else "notrunc"
                                tid = (
                                    f"n{n}-{tie_method}-{tie_pattern}-{strat_tag}"
                                    f"-{weight_type}-{trunc_tag}-s{seed}"
                                )
                                params.append(
                                    pytest.param(
                                        dict(
                                            n=n,
                                            tie_method=tie_method,
                                            tie_pattern=tie_pattern,
                                            stratified=stratified,
                                            weight_type=weight_type,
                                            left_truncation=left_truncation,
                                            seed=seed,
                                        ),
                                        id=tid,
                                    )
                                )
    return params


_SCENARIO_PARAMS = _build_scenario_params()


# ---------------------------------------------------------------------------
# Main parametrised test class
# ---------------------------------------------------------------------------
class TestToleranceProgression:
    """Parametrised over the full scenario grid."""

    @pytest.mark.parametrize("scenario", _SCENARIO_PARAMS)
    def test_scenario(self, scenario):
        report = run_scenario(**scenario)
        _all_reports.append(report)

        # Print detailed metrics (visible with -s)
        print(f"\nScenario: {report.label}")
        g = report.gradient
        h = report.hessian
        print(
            f"  Gradient:   max_abs={g.max_abs:.2e}  "
            f"max_rel={g.max_rel:.2e}  L2_rel={g.L2_rel:.2e}"
        )
        print(
            f"  Hessian:    max_abs={h.max_abs:.2e}  "
            f"max_rel={h.max_rel:.2e}  L2_rel={h.L2_rel:.2e}"
        )
        print(
            f"  LoglikSat:  abs={report.loglik_sat_abs:.2e}  "
            f"rel={report.loglik_sat_rel:.2e}"
        )

        tol_str = "  Tolerance: "
        for tol in TOLERANCE_GRID:
            tag = "pass" if report.tolerance_pass(tol) else "FAIL"
            tol_str += f" {tol:.0e}: {tag} "
        print(tol_str)

        # Hard-assert at regression-guard level
        assert report.gradient.passes(REGRESSION_TOL), (
            f"Gradient regression failure: "
            f"max_abs={g.max_abs:.2e}, max_rel={g.max_rel:.2e}"
        )
        assert report.hessian.passes(REGRESSION_TOL), (
            f"Hessian regression failure: "
            f"max_abs={h.max_abs:.2e}, max_rel={h.max_rel:.2e}"
        )
        assert report.loglik_sat_abs <= REGRESSION_TOL or report.loglik_sat_rel <= REGRESSION_TOL, (
            f"LoglikSat regression failure: "
            f"abs={report.loglik_sat_abs:.2e}, rel={report.loglik_sat_rel:.2e}"
        )


# ---------------------------------------------------------------------------
# Summary table (run last)
# ---------------------------------------------------------------------------
def test_summary_table():
    """Print consolidated tolerance progression summary.

    This test should be run after the parametrised tests so that
    _all_reports is populated. If run in isolation it re-runs a
    representative subset.
    """
    reports = _all_reports

    # If no reports collected yet, run a representative subset
    if not reports:
        subset = [
            dict(n=100, tie_method="efron", tie_pattern="moderate",
                 stratified=False, weight_type="random", left_truncation=True, seed=42),
            dict(n=100, tie_method="breslow", tie_pattern="heavy",
                 stratified=True, weight_type="random_zeros", left_truncation=False, seed=42),
            dict(n=500, tie_method="efron", tie_pattern="none",
                 stratified=False, weight_type="uniform", left_truncation=False, seed=42),
            dict(n=20, tie_method="efron", tie_pattern="heavy",
                 stratified=True, weight_type="random", left_truncation=True, seed=42),
        ]
        for s in subset:
            reports.append(run_scenario(**s))

    # --- Header ---
    tol_hdr = "  ".join(f"{t:.0e}" for t in TOLERANCE_GRID)
    print(f"\n{'='*120}")
    print("TOLERANCE PROGRESSION SUMMARY")
    print(f"{'='*120}")
    print(f"  {'Scenario':<60s}  {tol_hdr}")
    print(f"  {'-'*60}  " + "  ".join("-" * 5 for _ in TOLERANCE_GRID))

    for r in reports:
        print(r.summary_line())

    # --- Tightest universal tolerance ---
    tightest = None
    for tol in TOLERANCE_GRID:
        if all(r.tolerance_pass(tol) for r in reports):
            tightest = tol
    print(f"\n  Tightest universal tolerance: {tightest:.0e}" if tightest else
          "\n  No universal tolerance found in grid")

    # --- Worst cases ---
    worst_grad = max(reports, key=lambda r: r.gradient.max_rel)
    worst_hess = max(reports, key=lambda r: r.hessian.max_rel)
    worst_sat = max(reports, key=lambda r: r.loglik_sat_rel)

    print(f"\n  Worst gradient  max_rel: {worst_grad.gradient.max_rel:.2e}  ({worst_grad.label})")
    print(f"  Worst hessian   max_rel: {worst_hess.hessian.max_rel:.2e}  ({worst_hess.label})")
    print(f"  Worst loglik_sat    rel: {worst_sat.loglik_sat_rel:.2e}  ({worst_sat.label})")
    print(f"{'='*120}\n")

    # Informational only: no assertions here.


# ---------------------------------------------------------------------------
# Tight-tolerance probes (xfail, report-only)
# ---------------------------------------------------------------------------
_TIGHT_SCENARIOS = [
    dict(n=100, tie_method="efron", tie_pattern="none",
         stratified=False, weight_type="uniform", left_truncation=False, seed=42),
    dict(n=100, tie_method="breslow", tie_pattern="moderate",
         stratified=False, weight_type="random", left_truncation=True, seed=42),
    dict(n=100, tie_method="efron", tie_pattern="heavy",
         stratified=True, weight_type="random_zeros", left_truncation=True, seed=42),
    dict(n=500, tie_method="efron", tie_pattern="moderate",
         stratified=False, weight_type="random", left_truncation=False, seed=42),
]

_TIGHT_TOLS = [1e-13, 1e-14, 1e-15]


class TestTightTolerances:
    """Probe ultra-tight tolerances. Marked xfail so they report but don't break CI."""

    @pytest.mark.parametrize(
        "scenario",
        [pytest.param(s, id=f"n{s['n']}-{s['tie_method']}-{s['tie_pattern']}")
         for s in _TIGHT_SCENARIOS],
    )
    @pytest.mark.parametrize("tight_tol", _TIGHT_TOLS, ids=[f"tol{t}" for t in _TIGHT_TOLS])
    @pytest.mark.xfail(strict=False, reason="Ultra-tight tolerance probe")
    def test_tight_tolerance(self, scenario, tight_tol):
        report = run_scenario(**scenario)

        g = report.gradient
        h = report.hessian
        print(f"\nTight probe: {report.label}  tol={tight_tol:.0e}")
        print(
            f"  Gradient:   max_abs={g.max_abs:.2e}  max_rel={g.max_rel:.2e}"
        )
        print(
            f"  Hessian:    max_abs={h.max_abs:.2e}  max_rel={h.max_rel:.2e}"
        )
        print(
            f"  LoglikSat:  abs={report.loglik_sat_abs:.2e}  "
            f"rel={report.loglik_sat_rel:.2e}"
        )

        assert report.tolerance_pass(tight_tol), (
            f"Did not pass at tol={tight_tol:.0e}: "
            f"grad_max_rel={g.max_rel:.2e}, hess_max_rel={h.max_rel:.2e}, "
            f"sat_rel={report.loglik_sat_rel:.2e}"
        )
