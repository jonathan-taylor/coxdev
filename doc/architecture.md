# coxdev Architecture

## File Structure (Adapter Pattern)

```
coxdev/
├── include/
│   └── coxdev.hpp              # Canonical header-only library (single source of truth)
├── src/
│   └── coxdev_py.cpp           # Python pybind11 bindings
├── R_pkg/coxdev/
│   ├── src/
│   │   └── coxdev_r.cpp        # R Rcpp bindings
│   └── inst/include/
│       └── coxdev.hpp → symlink to ../../../../include/coxdev.hpp
└── setup.py                    # Python build (uses src/coxdev_py.cpp + include/)
```

**Key Design Principles:**

1. **Single Source of Truth**: `include/coxdev.hpp` is the only C++ implementation
2. **Zero-Copy Bindings**: R uses `Eigen::Map`, Python uses pybind11 array mapping
3. **Conditional Namespace**: `#ifdef GLMNET_INTERFACE` switches to `glmnetpp::coxdev::`
4. **Independence**: coxdev has no dependencies on glmnet; dependency is one-way

**To use in glmnet**: Copy `include/coxdev.hpp` unchanged, define `GLMNET_INTERFACE` before including.

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      coxdev C++ Core                            │
│                                                                 │
│  Layer 1: Data Preprocessing (computed once per dataset)        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CoxPreprocessed: event_order, risk sets, tie groups,      │  │
│  │                  Efron scaling, start/stop mappings       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Layer 2: IRLS State (recomputed once per outer iteration)      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CoxIRLSState: exp(η), risk_sums, working weights w,       │  │
│  │               working response z                          │  │
│  │                                                           │  │
│  │   - recompute_outer(η)           [O(15n), once per outer] │  │
│  │   - get_working_weights()        [O(1), accessor]         │  │
│  │   - get_working_response()       [O(1), accessor]         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Layer 3: Coordinate Descent Primitives (per inner iteration)   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │   - coordinate_gradient(j, x_j)  [O(n), uses cached w/z]  │  │
│  │   - update_residual(j, δ, x_j)   [O(n), incremental]      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Convenience API (backward compatible, stateless)               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ cox_dev(η) → full recompute                [STATELESS]    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │                           │
              ▼                           ▼
         R (Rcpp/XPtr)             Python (pybind11)
```

## Integration with glmnetpp

glmnetpp owns the coordinate descent loop structure; coxdev provides Cox-specific primitives:

```
glmnetpp (ElnetPointInternalCox)              coxdev
┌─────────────────────────────────┐          ┌─────────────────────────┐
│ Outer IRLS loop                 │          │ CoxIRLSState            │
│   │                             │  ──────► │   recompute_outer(η)    │
│   ▼                             │          │   get_w(), get_z()      │
│ Inner CD loop                   │          │                         │
│   for j in 1..p:                │  ──────► │   coordinate_grad(j,xj) │
│     update β_j                  │  ──────► │   update_residual(j,δ)  │
│   until converged               │          │                         │
└─────────────────────────────────┘          └─────────────────────────┘
```

**Key principle**: coxdev remains penalty-agnostic. It provides Cox likelihood primitives; glmnetpp handles elastic net penalties, active sets, and convergence.

## Dependency Direction

```
coxdev (independent)              glmnet/glmnetpp (depends on coxdev)
┌─────────────────────────┐       ┌─────────────────────────────────┐
│ CoxPreprocessed         │       │ ElnetPointInternalCox*          │
│ CoxWorkspace            │ ───►  │   - holds CoxIRLSState member   │
│ StratifiedCoxData       │ copy  │   - calls its methods           │
│ CoxIRLSState            │       │                                 │
└─────────────────────────┘       └─────────────────────────────────┘
```

## Performance Model

| Implementation | Inner Loop Cost per λ |
|----------------|----------------------|
| Stateless (original) | O(15 × t × k × n) where t=outer iters, k=CD iters |
| Stateful (CoxIRLSState) | O(t × k × n) + O(t × 15n) |

For typical convergence (t~5 outer, k~10 inner), stateful provides ~10-15x speedup for the inner loop.
