# Per-call times of CoxDev.jl (the library at ENV["COXDEV_C_LIB"]):
#   deviance (fresh eta each call), the information product I*v, deviance
#   followed by I*v at the same eta, and the raw ccall with preallocated
#   buffers (no wrapper conversions), plus a copy of n doubles for scale.
# Usage: julia --project=CoxDev.jl bench_julia.jl <data dir> <label>
using CoxDev, JSON, Statistics
const C = CoxDev

dir, label = ARGS[1], ARGS[2]

function per_call(f, reps)
    f(); f()
    ts = Float64[]
    for _ in 1:reps
        t = time_ns(); f(); push!(ts, (time_ns() - t) / 1e9)
    end
    median(ts)
end

res = Dict{String,Any}()
for n in (10^4, 10^5, 10^6)
    a = Vector{Float64}(undef, 6n)
    read!(joinpath(dir, "data_$n.bin"), a)
    a = reshape(a, n, 6)
    event, status, eta1, eta2, v, w = (a[:, j] for j in 1:6)
    cd = CoxDeviance(event, Int.(status); weight=w, ties=:efron)
    reps = n <= 10^5 ? 200 : 30
    etas = (eta1, eta2); k = Ref(1)
    flip() = (k[] = 3 - k[]; etas[k[]])

    t_dev = per_call(() -> deviance(cd, flip(); weight=w), reps)
    I = information(cd, eta1; weight=w)
    y = similar(v)
    t_mv = per_call(() -> C.LinearMaps.mul!(y, I, v), reps)
    t_both = per_call(reps) do
        e = flip()
        deviance(cd, e; weight=w)
        information(cd, e; weight=w) * v
    end

    # the raw ccall with preallocated buffers: the C library's own time
    dev, sat = Ref{Cdouble}(0), Ref{Cdouble}(0)
    g, h = similar(v), similar(v)
    fdev = C._sym(:coxdev_deviance)
    fmv = C._sym(:coxdev_hessian_matvec)
    raw_dev() = ccall(fdev, Cint,
                      (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}),
                      cd.ptr, flip(), w, dev, g, h, sat)
    raw_mv() = ccall(fmv, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                     cd.ptr, eta1, w, v, y)
    t_raw_dev = per_call(raw_dev, reps)
    t_raw_mv = per_call(raw_mv, reps)

    # one copy of n doubles, for scale (the C API copies eta and the weights in,
    # compares the weights, and copies the gradient and diagonal Hessian out)
    buf = similar(v)
    t_copy = per_call(() -> copyto!(buf, v), reps)

    res[string(n)] = Dict("deviance" => t_dev, "matvec" => t_mv, "deviance_then_matvec" => t_both,
                          "raw_deviance" => t_raw_dev, "raw_matvec" => t_raw_mv, "copy_n" => t_copy)
end
println(JSON.json(Dict(label => res)))
