# repeated deviance at the same eta: Julia-cache hit vs C-cache hit vs full evaluation
using CoxDev, Statistics
const C = CoxDev
function per_call(f, reps); f(); f(); median([(t = time_ns(); f(); (time_ns() - t) / 1e9) for _ in 1:reps]); end
for n in (10^4, 10^5, 10^6)
    a = Vector{Float64}(undef, 6n); read!("data_$n.bin", a); a = reshape(a, n, 6)
    event, status, eta1, eta2, v, w = (a[:, j] for j in 1:6)
    cd = CoxDeviance(event, Int.(status); weight=w)
    reps = n <= 10^5 ? 200 : 30
    t_julia_hit = per_call(() -> deviance(cd, eta1; weight=w), reps)
    # bypass the Julia cache: alternate a copy of the weights' identity is irrelevant (contents);
    # call the C library directly at the same eta (C-cache hit)
    dev, sat = Ref{Cdouble}(0), Ref{Cdouble}(0); g, h = similar(v), similar(v)
    f = C._sym(:coxdev_deviance)
    t_c_hit = per_call(() -> ccall(f, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ref{Cdouble}), cd.ptr, eta1, w, dev, g, h, sat), reps)
    k = Ref(1); etas = (eta1, eta2)
    flip() = (k[] = 3 - k[]; etas[k[]])
    t_miss = per_call(() -> deviance(cd, flip(); weight=w), reps)
    println("n=$n  full evaluation $(round(t_miss*1e3, digits=3)) ms;  C-cache hit $(round(t_c_hit*1e3, digits=3)) ms;  Julia-cache hit $(round(t_julia_hit*1e3, digits=3)) ms")
end
