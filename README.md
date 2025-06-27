# coxdev

A high-performance Python library for computing Cox proportional hazards model deviance, gradients, and Hessian information matrices. Built with C++ and Eigen for optimal performance, this library provides efficient survival analysis computations with support for different tie-breaking methods.

## Features

- **High Performance**: C++ implementation with Eigen linear algebra library
- **Comprehensive Support**: Handles both Efron and Breslow tie-breaking methods
- **Left-Truncated Data**: Support for left-truncated survival data
- **Efficient Computations**: Optimized algorithms for deviance, gradient, and Hessian calculations
- **Memory Efficient**: Uses linear operators for large-scale computations
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Installation

### Prerequisites

This package requires the Eigen C++ library headers. The Eigen library is included as a git submodule.

1. **Initialize Eigen submodule**: The Eigen library is included as a git submodule. Make sure it's initialized:
   ```bash
   git submodule update --init --recursive
   ```

2. **Check Eigen availability**: Run the check script to verify Eigen headers are available:
   ```bash
   python check_eigen.py
   ```

### Standard Installation

```bash
pip install .
```

### With Custom Eigen Path

If you have Eigen installed elsewhere, you can specify its location:
```bash
env EIGEN_LIBRARY_PATH=/path/to/eigen pip install .
```

### Development Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from coxdev import CoxDeviance

# Generate sample survival data
n_samples = 1000
event_times = np.random.exponential(1.0, n_samples)
status = np.random.binomial(1, 0.7, n_samples)  # 70% events, 30% censored
linear_predictor = np.random.normal(0, 1, n_samples)

# Create CoxDeviance object
coxdev = CoxDeviance(event=event_times, status=status, tie_breaking='efron')

# Compute deviance and related quantities
result = coxdev(linear_predictor)

print(f"Deviance: {result.deviance:.4f}")
print(f"Saturated log-likelihood: {result.loglik_sat:.4f}")
print(f"Gradient norm: {np.linalg.norm(result.gradient):.4f}")
```

## Advanced Usage

### Left-Truncated Data

```python
# With start times (left-truncated data)
start_times = np.random.exponential(0.5, n_samples)
coxdev = CoxDeviance(
    event=event_times, 
    status=status, 
    start=start_times,
    tie_breaking='efron'
)
```

### Computing Information Matrix

```python
# Get information matrix as a linear operator
info_matrix = coxdev.information(linear_predictor)

# Matrix-vector multiplication
v = np.random.normal(0, 1, n_samples)
result_vector = info_matrix @ v

# For small problems, you can compute the full matrix
X = np.random.normal(0, 1, (n_samples, 10))
beta = np.random.normal(0, 1, 10)
eta = X @ beta

# Information matrix for coefficients: X^T @ I @ X
I = info_matrix @ X
information_matrix = X.T @ I
```

### Different Tie-Breaking Methods

```python
# Efron's method (default)
coxdev_efron = CoxDeviance(event=event_times, status=status, tie_breaking='efron')

# Breslow's method
coxdev_breslow = CoxDeviance(event=event_times, status=status, tie_breaking='breslow')
```

## API Reference

### CoxDeviance

The main class for computing Cox model quantities.

#### Parameters

- **event**: Event times (failure times) for each observation
- **status**: Event indicators (1 for event occurred, 0 for censored)
- **start**: Start times for left-truncated data (optional)
- **tie_breaking**: Method for handling tied event times ('efron' or 'breslow')

#### Methods

- **`__call__(linear_predictor, sample_weight=None)`**: Compute deviance and related quantities
- **`information(linear_predictor, sample_weight=None)`**: Get information matrix as linear operator

### CoxDevianceResult

Result object containing computation results.

#### Attributes

- **linear_predictor**: The linear predictor values used
- **sample_weight**: Sample weights used
- **loglik_sat**: Saturated log-likelihood value
- **deviance**: Computed deviance value
- **gradient**: Gradient of deviance with respect to linear predictor
- **diag_hessian**: Diagonal of Hessian matrix

## Performance

The library is optimized for performance:

- **C++ Implementation**: Core computations in C++ with Eigen
- **Memory Efficient**: Reuses buffers and uses linear operators
- **Vectorized Operations**: Leverages Eigen's optimized linear algebra
- **Minimal Python Overhead**: Heavy computations done in C++

## Building from Source

### Prerequisites

- Python 3.9+
- C++ compiler with C++17 support
- Eigen library headers
- pybind11

### Build Steps

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/jonathan-taylor/coxdev.git
   cd coxdev
   ```

2. Install build dependencies:
   ```bash
   pip install build wheel setuptools pybind11 numpy
   ```

3. Build the package:
   ```bash
   python -m build
   ```

### Building Wheels

For wheel building:
```bash
# Standard wheel build
python -m build

# With custom Eigen path
env EIGEN_LIBRARY_PATH=/path/to/eigen python -m build
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{coxdev2024,
  title={coxdev: High-performance Cox proportional hazards deviance computation},
  author={Taylor, Jonathan and Hastie, Trevor and Narasimhan, Balasubramanian},
  year={2024},
  url={https://github.com/jonathan-taylor/coxdev}
}
```

## Acknowledgments

- Built with [Eigen](http://eigen.tuxfamily.org/) for efficient linear algebra
- Uses [pybind11](https://pybind11.readthedocs.io/) for Python bindings
- Inspired by the R `glmnet` package for survival analysis
