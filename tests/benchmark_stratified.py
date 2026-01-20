"""
Benchmark: C++ vs Python Stratified Cox Implementation

Compares performance across different scenarios:
- Number of strata
- Number of observations
- With/without ties
- With/without start times
"""

import numpy as np
import time
from coxdev import StratifiedCoxDeviance, StratifiedCoxDevianceCpp


def benchmark_single(n, n_strata, n_calls=100, with_ties=False, with_start=False, seed=42):
    """Run a single benchmark comparison."""
    np.random.seed(seed)

    if with_ties:
        # Create data with many ties
        event = np.random.choice(np.arange(1, n // 10 + 1), n).astype(float)
    else:
        event = np.random.exponential(5, n)

    status = np.random.binomial(1, 0.7, n)
    strata = np.random.choice(n_strata, n)
    eta = np.random.randn(n) * 0.5
    weights = np.random.uniform(0.5, 2.0, n)

    if with_start:
        start = np.random.uniform(0, 2, n)
        event = start + np.abs(event)  # Ensure event > start
    else:
        start = None

    # Create objects (preprocessing time)
    t0 = time.perf_counter()
    py_cox = StratifiedCoxDeviance(event=event, status=status, strata=strata, start=start)
    py_init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpp_cox = StratifiedCoxDevianceCpp(event=event, status=status, strata=strata, start=start)
    cpp_init_time = time.perf_counter() - t0

    # Warm up
    py_cox(eta, weights)
    cpp_cox(eta, weights)

    # Benchmark deviance computation
    t0 = time.perf_counter()
    for _ in range(n_calls):
        py_result = py_cox(eta, weights)
    py_call_time = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        cpp_result = cpp_cox(eta, weights)
    cpp_call_time = (time.perf_counter() - t0) / n_calls

    # Benchmark information matvec
    v = np.random.randn(n)
    py_info = py_cox.information(eta, weights)
    cpp_info = cpp_cox.information(eta, weights)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = py_info @ v
    py_matvec_time = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = cpp_info @ v
    cpp_matvec_time = (time.perf_counter() - t0) / n_calls

    # Verify results match
    assert np.isclose(py_result.deviance, cpp_result.deviance, rtol=1e-10)

    return {
        'n': n,
        'n_strata': n_strata,
        'with_ties': with_ties,
        'with_start': with_start,
        'py_init_ms': py_init_time * 1000,
        'cpp_init_ms': cpp_init_time * 1000,
        'py_call_ms': py_call_time * 1000,
        'cpp_call_ms': cpp_call_time * 1000,
        'py_matvec_ms': py_matvec_time * 1000,
        'cpp_matvec_ms': cpp_matvec_time * 1000,
        'init_speedup': py_init_time / cpp_init_time if cpp_init_time > 0 else float('inf'),
        'call_speedup': py_call_time / cpp_call_time if cpp_call_time > 0 else float('inf'),
        'matvec_speedup': py_matvec_time / cpp_matvec_time if cpp_matvec_time > 0 else float('inf'),
    }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("BENCHMARK: C++ vs Python Stratified Cox Implementation")
    print("=" * 80)

    results = []

    # Varying number of observations with fixed strata
    print("\n### Varying n (observations), fixed 10 strata ###")
    print(f"{'n':>8} | {'Python (ms)':>12} | {'C++ (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)
    for n in [100, 500, 1000, 5000, 10000]:
        r = benchmark_single(n, n_strata=10, n_calls=50)
        results.append(r)
        print(f"{n:>8} | {r['py_call_ms']:>12.4f} | {r['cpp_call_ms']:>12.4f} | {r['call_speedup']:>8.2f}x")

    # Varying number of strata with fixed observations
    print("\n### Varying strata count, fixed n=1000 ###")
    print(f"{'strata':>8} | {'Python (ms)':>12} | {'C++ (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)
    for n_strata in [2, 5, 10, 50, 100, 200]:
        r = benchmark_single(n=1000, n_strata=n_strata, n_calls=50)
        results.append(r)
        print(f"{n_strata:>8} | {r['py_call_ms']:>12.4f} | {r['cpp_call_ms']:>12.4f} | {r['call_speedup']:>8.2f}x")

    # Many small strata (worst case for Python loop overhead)
    print("\n### Many small strata (5 obs each) ###")
    print(f"{'strata':>8} | {'Python (ms)':>12} | {'C++ (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)
    for n_strata in [10, 50, 100, 200, 500]:
        n = n_strata * 5
        r = benchmark_single(n=n, n_strata=n_strata, n_calls=50)
        results.append(r)
        print(f"{n_strata:>8} | {r['py_call_ms']:>12.4f} | {r['cpp_call_ms']:>12.4f} | {r['call_speedup']:>8.2f}x")

    # With ties (Efron)
    print("\n### With many ties (Efron), n=1000, 10 strata ###")
    r = benchmark_single(n=1000, n_strata=10, n_calls=50, with_ties=True)
    results.append(r)
    print(f"Python: {r['py_call_ms']:.4f} ms | C++: {r['cpp_call_ms']:.4f} ms | Speedup: {r['call_speedup']:.2f}x")

    # With start times (left truncation)
    print("\n### With start times (left truncation), n=1000, 10 strata ###")
    r = benchmark_single(n=1000, n_strata=10, n_calls=50, with_start=True)
    results.append(r)
    print(f"Python: {r['py_call_ms']:.4f} ms | C++: {r['cpp_call_ms']:.4f} ms | Speedup: {r['call_speedup']:.2f}x")

    # Information matrix matvec
    print("\n### Information matrix matvec, n=1000, 10 strata ###")
    r = benchmark_single(n=1000, n_strata=10, n_calls=50)
    print(f"Python: {r['py_matvec_ms']:.4f} ms | C++: {r['cpp_matvec_ms']:.4f} ms | Speedup: {r['matvec_speedup']:.2f}x")

    # Initialization time
    print("\n### Initialization (preprocessing) time, n=5000, 50 strata ###")
    r = benchmark_single(n=5000, n_strata=50, n_calls=10)
    print(f"Python: {r['py_init_ms']:.4f} ms | C++: {r['cpp_init_ms']:.4f} ms | Speedup: {r['init_speedup']:.2f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Compute average speedups
    call_speedups = [r['call_speedup'] for r in results]
    print(f"Call speedup range: {min(call_speedups):.2f}x - {max(call_speedups):.2f}x")
    print(f"Mean call speedup: {np.mean(call_speedups):.2f}x")

    return results


if __name__ == "__main__":
    run_benchmarks()
