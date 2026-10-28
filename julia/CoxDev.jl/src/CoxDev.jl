"""
    CoxDev

Cox proportional hazards deviance, gradient, information and score draws from
coxdev's C++ library, through its C API (`coxdev_c_api.h`). The same code as
coxdev's Python and R packages: the same numbers, and for `sample_score` the
same draws for the same seed (to rounding across platforms).

The shared library `libcoxdev_c` is found from the environment variable
`COXDEV_C_LIB` (its path), else from a build of `cpp/capi` in the coxdev
checkout this package sits in. A released package would get it from a
`coxdev_jll` binary package instead (see `README.md`).
"""
module CoxDev

using Libdl
using LinearMaps

export CoxDeviance, deviance, information, information_derivative, baseline_hazard, concordance,
       score_residuals, wild_score, sample_score, survfit, expected_events,
       martingale_residuals, deviance_residuals, CoxDevError

# ---------------------------------------------------------------------------
# the shared library

const _lib = Ref{Ptr{Cvoid}}(C_NULL)
const _syms = Dict{Symbol,Ptr{Cvoid}}()
const _syms_lock = ReentrantLock()

"""
    libpath()

The path of `libcoxdev_c`: `ENV["COXDEV_C_LIB"]`, else `cpp/capi/build` of the
coxdev checkout containing this package. (A `coxdev_jll` would replace this.)
"""
function libpath()
    p = get(ENV, "COXDEV_C_LIB", "")
    isempty(p) || return p
    build = normpath(joinpath(@__DIR__, "..", "..", "..", "cpp", "capi", "build"))
    if isdir(build)
        for f in readdir(build)
            startswith(f, "libcoxdev_c.") && endswith(f, "." * Libdl.dlext) && return joinpath(build, f)
        end
    end
    error("libcoxdev_c not found: set COXDEV_C_LIB, or build it with cpp/capi/build_local.sh")
end

function _sym(name::Symbol)
    lock(_syms_lock) do
        get!(_syms, name) do
            if _lib[] == C_NULL
                _lib[] = Libdl.dlopen(libpath())
                v = ccall(Libdl.dlsym(_lib[], :coxdev_api_version), Cint, ())
                v >= 3 || error("libcoxdev_c has C API version $v; CoxDev.jl needs 3 or later")
            end
            Libdl.dlsym(_lib[], name)
        end
    end
end

"""
    CoxDevError(code, msg)

An error reported by the coxdev C library (`code` < 0, see `coxdev_c_api.h`).
"""
struct CoxDevError <: Exception
    code::Int
    msg::String
end
Base.showerror(io::IO, e::CoxDevError) = print(io, "CoxDevError(", e.code, "): ", e.msg)

function _check(code::Cint)
    if code != 0
        msg = unsafe_string(ccall(_sym(:coxdev_last_error), Cstring, ()))
        throw(CoxDevError(Int(code), msg))
    end
    nothing
end

_dptr(::Nothing) = Ptr{Cdouble}(C_NULL)
_dptr(x) = x

# ---------------------------------------------------------------------------
# the model

"""
    CoxDeviance(event, status; start=nothing, weight=nothing, ties=:efron, strata=nothing)

A Cox model's data: failure or censoring times `event`, `status` (1 failure,
0 censored), optional entry times `start` (for `(start, event]` data), weights
and strata (any labels `sort` can order). `ties` is `:efron` or `:breslow`.

`weight` (default ones) are the object's weights. As in coxdev's Python and
R packages, a function called without `weight` uses the object's current
weights: these, or the last ones given to a function (the C object is
rebuilt when a call gives other weights, and keeps them; `concordance` does
not change them).

A `CoxDeviance` is not safe to use from several tasks at once without the
lock it holds (all methods take it); distinct objects are independent.
"""
mutable struct CoxDeviance
    ptr::Ptr{Cvoid}
    n::Int
    ties::Symbol
    event::Vector{Float64}
    status::Vector{Cint}
    start::Union{Nothing,Vector{Float64}}
    strata_codes::Union{Nothing,Vector{Cint}}
    strata_labels::Union{Nothing,Vector}
    lock::ReentrantLock
    # the last deviance call, reused while eta and the weight are unchanged
    # (as coxdev's Python CoxDeviance caches its last call)
    last_eta::Vector{Float64}
    last_weight::Vector{Float64}
    last::Any

    function CoxDeviance(event::AbstractVector, status::AbstractVector;
                         start=nothing, weight=nothing, ties::Symbol=:efron, strata=nothing)
        n = length(event)
        length(status) == n || throw(ArgumentError("event and status must have the same length"))
        all(s -> s == 0 || s == 1, status) || throw(ArgumentError("status must be binary"))
        ties in (:efron, :breslow) || throw(ArgumentError("ties must be :efron or :breslow"))
        ev = Vector{Float64}(event)
        st = Vector{Cint}(status)
        sta = start === nothing ? nothing : Vector{Float64}(start)
        sta === nothing || length(sta) == n || throw(ArgumentError("start must have length $n"))
        codes, labels = nothing, nothing
        if strata !== nothing
            length(strata) == n || throw(ArgumentError("strata must have length $n"))
            labels = sort(unique(strata))
            index = Dict(l => Cint(i - 1) for (i, l) in enumerate(labels))
            codes = Cint[index[s] for s in strata]
        end
        w = weight === nothing ? nothing : Vector{Float64}(weight)
        w === nothing || length(w) == n || throw(ArgumentError("weight must have length $n"))
        out = Ref{Ptr{Cvoid}}(C_NULL)
        _check(ccall(_sym(:coxdev_new), Cint,
                     (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Ref{Ptr{Cvoid}}),
                     n, _dptr(sta), ev, st, codes === nothing ? Ptr{Cint}(C_NULL) : codes, _dptr(w),
                     ties == :efron ? 1 : 0, out))
        cd = new(out[], n, ties, ev, st, sta, codes, labels, ReentrantLock(),
                 Float64[], Float64[], nothing)
        finalizer(cd) do x
            if x.ptr != C_NULL
                ccall(_sym(:coxdev_free), Cvoid, (Ptr{Cvoid},), x.ptr)
                x.ptr = C_NULL
            end
        end
        return cd
    end
end

Base.length(cd::CoxDeviance) = cd.n
Base.show(io::IO, cd::CoxDeviance) =
    print(io, "CoxDeviance(n=", cd.n, ", ties=:", cd.ties,
          cd.start === nothing ? "" : ", (start, event]",
          cd.strata_labels === nothing ? "" : ", $(length(cd.strata_labels)) strata", ")")

# Float64 vectors are passed to C as they are (no copy); anything else is
# converted once. The C library copies what it keeps.
_f64(x::Vector{Float64}) = x
_f64(x) = Vector{Float64}(x)

# weight = nothing means the handle's current weights (as in Python and R,
# and the C API's NULL), copied so that the deviance cache compares them
function _weight(cd::CoxDeviance, weight)
    if weight === nothing
        w = Vector{Float64}(undef, cd.n)
        lock(cd.lock) do
            _check(ccall(_sym(:coxdev_weight), Cint, (Ptr{Cvoid}, Ptr{Cdouble}), _live(cd), w))
        end
        return w
    end
    w = _f64(weight)
    length(w) == cd.n || throw(ArgumentError("weight must have length $(cd.n)"))
    w
end

function _eta(cd::CoxDeviance, eta)
    e = _f64(eta)
    length(e) == cd.n || throw(ArgumentError("linear predictor must have length $(cd.n)"))
    e
end

function _live(cd::CoxDeviance)
    cd.ptr == C_NULL && throw(ArgumentError("CoxDeviance has been freed"))
    cd.ptr
end

"""
    deviance(cd, eta; weight=nothing)

The deviance `2 (loglik_sat - loglik)` at the linear predictor `eta`, with its
gradient and diagonal Hessian in `eta`: a NamedTuple `(deviance, gradient,
diag_hessian, loglik_sat, loglik)`, with `loglik` the log partial likelihood
(R's `coxph` `loglik`). `weight` defaults to the object's current weights.

The result of the last call is reused while `eta` and `weight` are unchanged,
as in coxdev's Python package: do not modify its arrays.
"""
function deviance(cd::CoxDeviance, eta::AbstractVector; weight=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    lock(cd.lock) do
        if cd.last !== nothing && e == cd.last_eta && w == cd.last_weight
            return cd.last
        end
        dev, sat = Ref{Cdouble}(0), Ref{Cdouble}(0)
        g, h = Vector{Float64}(undef, cd.n), Vector{Float64}(undef, cd.n)
        _check(ccall(_sym(:coxdev_deviance), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}),
                     _live(cd), e, w, dev, g, h, sat))
        cd.last = (deviance=dev[], gradient=g, diag_hessian=h, loglik_sat=sat[],
                   loglik=sat[] - dev[] / 2)
        copyto!(resize!(cd.last_eta, cd.n), e)
        copyto!(resize!(cd.last_weight, cd.n), w)
        cd.last
    end
end

function _matvec!(y, cd::CoxDeviance, e, w, v)
    vv = _f64(v)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_hessian_matvec), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                     _live(cd), e, w, vv, y))
    end
    y
end

"""
    information(cd, eta; weight=nothing)

The information at `eta` — half the Hessian of the deviance — as a symmetric,
matrix-free `LinearMap` (each product costs `O(n)`): `X' * (I * X)` is the
information of the coefficients of a design `X`.
"""
function information(cd::CoxDeviance, eta::AbstractVector; weight=nothing)
    # copies, so that the operator stays at this eta whatever the caller does later
    e, w = copy(_eta(cd, eta)), copy(_weight(cd, weight))
    buf = Vector{Float64}(undef, cd.n)
    f! = (y, v) -> (y isa Vector{Float64} ? _matvec!(y, cd, e, w, v) : copyto!(y, _matvec!(buf, cd, e, w, v)))
    LinearMap{Float64}(f!, cd.n; issymmetric=true, ismutating=true)
end

function _dmatvec!(y, cd::CoxDeviance, e, w, d, v)
    vv = _f64(v)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_information_derivative_matvec), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                     _live(cd), e, w, d, vv, y))
    end
    y
end

"""
    information_derivative(cd, eta, direction; weight=nothing)

The derivative of the information at `eta` in the direction `direction` of the
linear predictor, `d/dt information(eta + t * direction)` at `t = 0`, as a
symmetric, matrix-free `LinearMap` (each product costs `O(n)`):
`X' * (dI * X)` is the derivative of the information of the coefficients of a
design `X`, as needed for the derivative of a log determinant (e.g. in REML),
with `direction` the derivative of the linear predictor. The derivative of the
Hessian of the deviance is twice it.
"""
function information_derivative(cd::CoxDeviance, eta::AbstractVector, direction::AbstractVector;
                                weight=nothing)
    length(direction) == cd.n || throw(ArgumentError("direction must have length $(cd.n)"))
    e, w, d = copy(_eta(cd, eta)), copy(_weight(cd, weight)), Vector{Float64}(direction)
    buf = Vector{Float64}(undef, cd.n)
    f! = (y, v) -> (y isa Vector{Float64} ? _dmatvec!(y, cd, e, w, d, v) : copyto!(y, _dmatvec!(buf, cd, e, w, d, v)))
    LinearMap{Float64}(f!, cd.n; issymmetric=true, ismutating=true)
end

"""
    baseline_hazard(cd, eta; weight=nothing)

Breslow's (or with `ties=:efron`, Efron's) estimate of the baseline hazard at
`eta`: a NamedTuple `(time, hazard, cumulative_hazard, stratum)`, with the
strata reported by their labels (0 without strata).
"""
function baseline_hazard(cd::CoxDeviance, eta::AbstractVector; weight=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    m = Ref{Csize_t}(0)
    sig = (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint})
    time = Float64[]; hazard = Float64[]; cum = Float64[]; stratum = Cint[]
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_baseline_hazard), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                     _live(cd), e, w, m, C_NULL, C_NULL, C_NULL, C_NULL))
        k = Int(m[])
        resize!(time, k); resize!(hazard, k); resize!(cum, k); resize!(stratum, k)
        _check(ccall(_sym(:coxdev_baseline_hazard), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                     _live(cd), e, w, m, time, hazard, cum, stratum))
    end
    labels = cd.strata_labels === nothing ? Int.(stratum) : cd.strata_labels[stratum .+ 1]
    (time=time, hazard=hazard, cumulative_hazard=cum, stratum=labels)
end

"""
    survfit(cd, eta; weight=nothing, new_eta=nothing, new_strata=nothing, center=nothing)

Survival curves at the fixed linear predictor `eta`: R's `survfit.coxph`
(`se.fit = FALSE`) of `coxph(y ~ offset(eta))`, with the ties of `cd`, as
`glmnet`'s `survfit.coxnet` computes the curves of a fitted model. A
NamedTuple `(time, stratum, n_risk, n_event, n_censor, cumulative_hazard,
survival)`, laid out as `survfit`'s: a row per distinct event time (of
failures and censorings) in increasing time within each stratum, with the
weight at risk and the weights of the failures and censorings there, and a
column of `cumulative_hazard` and `survival` per curve (`NaN` on the rows of
other strata).

`weight` defaults to the current weights of `cd` (see `CoxDeviance`), so
that the `CoxDeviance` of a fitted model gives its curves. `new_eta` are the
linear predictors of the subjects to give curves of (with `new_strata` when
`cd` has strata); without it, one curve (in each stratum) at `center`,
which defaults to the weighted mean of `eta`.
"""
function survfit(cd::CoxDeviance, eta::AbstractVector; weight=nothing, new_eta=nothing,
                 new_strata=nothing, center=nothing)
    e = _eta(cd, eta)
    w = Vector{Float64}(undef, cd.n)
    m = Ref{Csize_t}(0)
    time = Float64[]; n_risk = Float64[]; n_event = Float64[]; n_censor = Float64[]
    H = Float64[]; stratum = Cint[]
    lock(cd.lock) do
        copyto!(w, _weight(cd, weight))
        _check(ccall(_sym(:coxdev_survival_table), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble},
                      Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                     _live(cd), e, w, m, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL))
        k = Int(m[])
        for v in (time, n_risk, n_event, n_censor, H)
            resize!(v, k)
        end
        resize!(stratum, k)
        _check(ccall(_sym(:coxdev_survival_table), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble},
                      Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                     _live(cd), e, w, m, time, n_risk, n_event, n_censor, H, stratum))
    end
    c = center === nothing ? sum(w .* e) / sum(w) : Float64(center)
    new = new_eta === nothing ? [c] : Vector{Float64}(new_eta)
    # for eta as given, H exp(new_eta), formed as (H exp(c)) exp(new_eta - c)
    cum = (H .* exp(c)) * transpose(exp.(new .- c))
    labels = cd.strata_labels === nothing ? Int.(stratum) : cd.strata_labels[stratum .+ 1]
    if cd.strata_labels !== nothing && new_eta !== nothing
        new_strata === nothing &&
            throw(ArgumentError("new_strata is needed for the curves of a stratified model"))
        length(new_strata) == length(new) ||
            throw(ArgumentError("new_strata and new_eta have different lengths"))
        all(s -> s in cd.strata_labels, new_strata) ||
            throw(ArgumentError("new_strata contains labels not seen in the training data"))
        for j in eachindex(new), i in eachindex(labels)
            labels[i] == new_strata[j] || (cum[i, j] = NaN)
        end
    end
    (time=time, stratum=labels, n_risk=n_risk, n_event=n_event, n_censor=n_censor,
     cumulative_hazard=cum, survival=exp.(-cum))
end

"""
    expected_events(cd, eta; weight=nothing)

The expected number of events of each observation at `eta`: R's
`predict(fit, type = "expected")` of a `coxph` fit. `weight` defaults to
the object's current weights.
"""
function expected_events(cd::CoxDeviance, eta::AbstractVector; weight=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    out = Vector{Float64}(undef, cd.n)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_expected_events), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                     _live(cd), e, w, out))
    end
    out
end

"""
    martingale_residuals(cd, eta; weight=nothing)

The martingale residuals at `eta`, `status - expected_events`: R's
`residuals(fit, type = "martingale")` of a `coxph` fit. `weight` defaults to
the object's current weights.
"""
martingale_residuals(cd::CoxDeviance, eta::AbstractVector; weight=nothing) =
    cd.status .- expected_events(cd, eta; weight=weight)

"""
    deviance_residuals(cd, eta; weight=nothing)

The deviance residuals at `eta`, `sign(M) sqrt(-2 (M + status log(status -
M)))` of the martingale residuals `M`: R's `residuals(fit, type =
"deviance")` of a `coxph` fit. `weight` defaults to the object's current
weights. A failure of zero weight with no hazard up to its time (its
expected events 0, so `M = 1`) has an infinite deviance residual.
"""
function deviance_residuals(cd::CoxDeviance, eta::AbstractVector; weight=nothing)
    M = martingale_residuals(cd, eta; weight=weight)
    d = [-2 * (M[i] + (cd.status[i] > 0 ? log(max(cd.status[i] - M[i], 0.0)) : 0.0))
         for i in eachindex(M)]
    sign.(M) .* sqrt.(max.(d, 0.0))
end

"""
    concordance(cd, pred; weight=nothing)

Harrell's concordance index (C-index) of the risk score `pred` (a vector, or a
matrix with one score per column) for the data of `cd`, within strata;
`NaN` when no pair is comparable. As R's `survival::concordance(y ~ -pred)`,
and the Python package's `CoxDeviance.c_index`.
"""
function concordance(cd::CoxDeviance, pred::AbstractVector; weight=nothing)
    p, w = _eta(cd, pred), _weight(cd, weight)
    val = Ref{Cdouble}(0)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_handle_concordance), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}),
                     _live(cd), p, w, C_NULL, C_NULL, val))
    end
    val[]
end
concordance(cd::CoxDeviance, pred::AbstractMatrix; weight=nothing) =
    [concordance(cd, view(pred, :, j); weight=weight) for j in axes(pred, 2)]

"""
    concordance(pred, event, status; start=nothing, weight=nothing)

Harrell's C-index of `pred` for unstratified `(start, event]` data, without
building a `CoxDeviance`.
"""
function concordance(pred::AbstractVector, event::AbstractVector, status::AbstractVector;
                     start=nothing, weight=nothing)
    n = length(pred)
    length(event) == n == length(status) || throw(ArgumentError("pred, event and status must have the same length"))
    p, ev, st = Vector{Float64}(pred), Vector{Float64}(event), Vector{Cint}(status)
    sta = start === nothing ? nothing : Vector{Float64}(start)
    w = weight === nothing ? nothing : Vector{Float64}(weight)
    val = Ref{Cdouble}(0)
    _check(ccall(_sym(:coxdev_concordance), Cint,
                 (Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}),
                 n, p, _dptr(sta), ev, st, _dptr(w), C_NULL, C_NULL, val))
    val[]
end

"""
    score_residuals(cd, eta, X; weight=nothing)

The score residuals of the columns of the design `X` (`n × p`) at `eta`: row
`i` is observation `i`'s contribution to the score `X' * (-gradient / 2)`,
with Efron's weights for ties if `ties=:efron` (Lin and Wei, 1989).
"""
function score_residuals(cd::CoxDeviance, eta::AbstractVector, X::AbstractMatrix; weight=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    size(X, 1) == cd.n || throw(ArgumentError("X must have $(cd.n) rows"))
    Xd = Matrix{Float64}(X)
    out = Matrix{Float64}(undef, size(Xd)...)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_score_residuals), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}),
                     _live(cd), e, w, Xd, size(Xd, 2), out))
    end
    out
end

"""
    wild_score(cd, eta, multipliers; weight=nothing)

The wild draw of the score for `multipliers` (a vector, or a matrix with one
draw per column): `r` with `X' * r = Σ_i multipliers[i] U_i`, for the score
residuals `U_i`. With multipliers all one it is `-gradient / 2`.
"""
function wild_score(cd::CoxDeviance, eta::AbstractVector, multipliers::AbstractVector; weight=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    g = Vector{Float64}(multipliers)
    length(g) == cd.n || throw(ArgumentError("multipliers must have length $(cd.n)"))
    out = Vector{Float64}(undef, cd.n)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_wild_score), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                     _live(cd), e, w, g, out))
    end
    out
end
wild_score(cd::CoxDeviance, eta::AbstractVector, G::AbstractMatrix; weight=nothing) =
    reduce(hcat, [wild_score(cd, eta, view(G, :, j); weight=weight) for j in axes(G, 2)])

"""
    sample_score(cd, eta; weight=nothing, method="multinomial", n_draws=nothing, seed=nothing)

Draws of the score at `eta`, on the scale of `-gradient / 2` (so `X' * r` is
a draw of the score of the coefficients of `X`). `method` is
`"multinomial"`, `"bernoulli"`, `"fisher"` (model-based; need equal positive
weights) or `"wild"`. One draw (a vector) if `n_draws` is `nothing`, else an
`n × n_draws` matrix. The same `seed` gives the same draws, here and in
coxdev's Python and R packages: the same random numbers on every platform, and
draws that agree to rounding (they go through `exp`, `log`, `sin` and `cos`,
whose last bits can differ between math libraries); `seed = nothing` draws
one.
"""
function sample_score(cd::CoxDeviance, eta::AbstractVector; weight=nothing,
                      method::AbstractString="multinomial", n_draws=nothing, seed=nothing)
    e, w = _eta(cd, eta), _weight(cd, weight)
    k = n_draws === nothing ? 1 : Int(n_draws)
    s = seed === nothing ? rand(UInt64) : UInt64(seed)
    out = Matrix{Float64}(undef, cd.n, k)
    lock(cd.lock) do
        _check(ccall(_sym(:coxdev_sample_score), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cstring, Csize_t, UInt64, Ptr{Cdouble}),
                     _live(cd), e, w, method, k, s, out))
    end
    n_draws === nothing ? vec(out) : out
end

end # module
