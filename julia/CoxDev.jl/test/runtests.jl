using CoxDev
using CoxDev: CoxDevError
using JSON
using LinearAlgebra
using Test

const FIXTURES = joinpath(@__DIR__, "fixtures")

# JSON null -> NaN; a vector of columns -> a matrix
fl(x) = Float64[v === nothing ? NaN : v for v in x]
mat(x) = reduce(hcat, [fl(c) for c in x])
labels(x) = x === nothing ? nothing : [v isa AbstractString ? String(v) : Int(v) for v in x]

# the same C++ underneath: agreement to rounding
approx(a, b) = isapprox(a, b; rtol=1e-12, atol=1e-13)

function load(name)
    JSON.parse(read(joinpath(FIXTURES, name), String))
end

function build(inp)
    CoxDeviance(fl(inp.event), Int.(inp.status);
                start=inp.start === nothing ? nothing : fl(inp.start),
                weight=inp.weight === nothing ? nothing : fl(inp.weight),
                ties=Symbol(inp.ties), strata=labels(inp.strata))
end

wt(inp) = inp.weight === nothing ? nothing : fl(inp.weight)

@testset "CoxDev.jl" begin

    files = sort(filter(f -> endswith(f, ".json"), readdir(FIXTURES)))
    @test length(files) == 36

    @testset "against Python coxdev: $(replace(f, ".json" => ""))" for f in files
        d = load(f)
        inp, out = d.inputs, d.outputs
        cd = build(inp)
        eta, w = fl(inp.eta), wt(inp)

        r = deviance(cd, eta; weight=w)
        @test approx(r.deviance, out.deviance)
        @test approx(r.loglik_sat, out.loglik_sat)
        @test approx(r.gradient, fl(out.gradient))
        @test approx(r.diag_hessian, fl(out.diag_hessian))
        # weight = nothing means ones, as in Python, whatever the construction weights
        @test approx(deviance(cd, eta).deviance, out.deviance_default_weight)

        I = information(cd, eta; weight=w)
        V = mat(inp.V)
        @test approx(Matrix(I * V), mat(out.information_V))
        @test issymmetric(I)

        dI = information_derivative(cd, eta, fl(inp.D); weight=w)
        @test approx(Matrix(dI * V), mat(out.information_derivative_V))
        @test issymmetric(dI)

        bh = baseline_hazard(cd, eta; weight=w)
        @test approx(bh.time, fl(out.bh_time))
        @test approx(bh.hazard, fl(out.bh_hazard))
        @test approx(bh.cumulative_hazard, fl(out.bh_cumulative_hazard))
        @test bh.stratum == labels(out.bh_stratum)

        @test approx(concordance(cd, eta; weight=w), out.c_index)
        if inp.strata === nothing
            # the free function, for unstratified data
            @test approx(concordance(eta, fl(inp.event), Int.(inp.status);
                                     start=inp.start === nothing ? nothing : fl(inp.start),
                                     weight=w), out.c_index)
        end

        @test approx(score_residuals(cd, eta, mat(inp.X); weight=w), mat(out.score_residuals))
        @test approx(wild_score(cd, eta, fl(inp.G); weight=w), fl(out.wild_score))

        for (m, draws) in pairs(out.sample_score)
            S = sample_score(cd, eta; weight=w, method=String(m), n_draws=inp.n_draws, seed=UInt64(inp.seed))
            # the same seed gives the same draws as Python's, to rounding: the
            # random numbers are the same on every platform, but the draws go
            # through exp, log, sin and cos, whose last bits differ between
            # math libraries (the fixtures are from macOS, CI is Linux)
            @test approx(S, mat(draws))
        end
    end

    @testset "finite differences" begin
        d = load("efron_startstop_int_random.json")
        inp = d.inputs
        cd = build(inp)
        eta, w = fl(inp.eta), wt(inp)
        r = deviance(cd, eta; weight=w)
        h = 1e-6
        n = length(eta)
        for i in (1, 7, n ÷ 2, n)
            e = zeros(n); e[i] = h
            fd = (deviance(cd, eta .+ e; weight=w).deviance - deviance(cd, eta .- e; weight=w).deviance) / (2h)
            @test isapprox(fd, r.gradient[i]; rtol=1e-6, atol=1e-7)
        end
        # the information is half the Hessian of the deviance: I v = (g(eta + h v) - g(eta - h v)) / (4h)
        I = information(cd, eta; weight=w)
        v = fl(inp.V[1])
        fd = (deviance(cd, eta .+ h .* v; weight=w).gradient .- deviance(cd, eta .- h .* v; weight=w).gradient) ./ (4h)
        @test isapprox(I * v, fd; rtol=1e-5, atol=1e-6)
        # and its diagonal is half the diagonal Hessian
        @test isapprox(diag(Matrix(I)), r.diag_hessian ./ 2; rtol=1e-10)
        # the derivative of the information: (I(eta + h d) v - I(eta - h d) v) / (2h),
        # symmetric in d and v
        d = fl(inp.D)
        dI = information_derivative(cd, eta, d; weight=w)
        fd = (information(cd, eta .+ h .* d; weight=w) * v .- information(cd, eta .- h .* d; weight=w) * v) ./ (2h)
        @test isapprox(dI * v, fd; rtol=1e-5, atol=1e-6)
        @test isapprox(dI * v, information_derivative(cd, eta, v; weight=w) * d; rtol=1e-10, atol=1e-12)
    end

    @testset "errors" begin
        @test_throws ArgumentError CoxDeviance([1.0, 2.0], [1, 2])          # status not binary
        @test_throws ArgumentError CoxDeviance([1.0, 2.0], [1])             # lengths
        @test_throws ArgumentError CoxDeviance([1.0, 2.0], [1, 0]; ties=:exact)
        cd = CoxDeviance([1.0, 2.0, 3.0], [1, 0, 1])
        @test_throws ArgumentError deviance(cd, [0.0, 0.0])                 # wrong length
        # an error from the C library
        err = try
            sample_score(cd, zeros(3); method="nonsense", seed=1)
            nothing
        catch e
            e
        end
        @test err isa CoxDevError
        @test err.code < 0
        @test occursin("method", lowercase(err.msg)) || !isempty(err.msg)
        # the model-based draws need equal weights
        @test_throws CoxDevError sample_score(cd, zeros(3); weight=[1.0, 2.0, 3.0], method="multinomial", seed=1)
    end

    @testset "caching of the last evaluation" begin
        d = load("efron_startstop_str_random.json")
        inp, out = d.inputs, d.outputs
        cd = build(inp)
        eta, w = fl(inp.eta), wt(inp)
        r1 = deviance(cd, eta; weight=w)
        # other calls at the same eta leave the cached evaluation valid (C and Julia caches)
        information(cd, eta; weight=w) * fl(inp.V[1])
        sample_score(cd, eta; weight=w, method="wild", n_draws=2, seed=1)
        score_residuals(cd, eta, mat(inp.X); weight=w)
        r2 = deviance(cd, eta; weight=w)
        @test r2 === r1
        @test approx(r2.deviance, out.deviance)
        # a different weight, then back: recomputed, same answer
        @test approx(deviance(cd, eta).deviance, out.deviance_default_weight)
        @test approx(deviance(cd, eta; weight=w).gradient, fl(out.gradient))
        # eta modified in place: the cache compares contents, so it recomputes
        e = copy(eta)
        r3 = deviance(cd, e; weight=w)
        e .+= 0.1
        r4 = deviance(cd, e; weight=w)
        @test r4.deviance != r3.deviance
        @test approx(r4.deviance, deviance(build(inp), e; weight=w).deviance)
        # an information operator stays at its eta when the caller's array changes
        e = copy(eta)
        I = information(cd, e; weight=w)
        v = fl(inp.V[2])
        Iv = I * v
        e .+= 1.0
        @test approx(I * v, Iv)
    end

    @testset "GC stress" begin
        d = load("efron_right_str_zero.json")
        inp = d.inputs
        eta, w = fl(inp.eta), wt(inp)
        dev = deviance(build(inp), eta; weight=w).deviance
        ok = true
        for k in 1:10_000
            cd = build(inp)
            k % 1000 == 0 && (ok &= approx(deviance(cd, eta; weight=w).deviance, dev))
            k % 2000 == 0 && GC.gc()
        end
        GC.gc()
        @test ok
    end
end
