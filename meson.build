project('coxdev', 'cpp',
  # python -m setuptools_scm reads [tool.setuptools_scm] in pyproject.toml
  # (only v* tags); get_version() alone would not
  version: run_command('python', '-m', 'setuptools_scm', check: true).stdout().strip(),
  default_options: ['cpp_std=c++17', 'warning_level=3'])

py = import('python').find_installation(pure: false)

# Eigen is vendored in src/eigen
inc = include_directories('eigen', 'cpp/include')

# Pybind11 dependency
pybind11_dep = dependency('pybind11', required: false)

if not pybind11_dep.found()
    pybind11_incdir = run_command(py,
      ['-c', 'import pybind11; print(pybind11.get_include())'],
      check: true
    ).stdout().strip()
    pybind11_dep = declare_dependency(include_directories: include_directories(pybind11_incdir))
endif

python_sources = [
  'python/coxdev/__init__.py',
  'python/coxdev/base.py',
  'python/coxdev/jax.py',
]

py.install_sources(
  python_sources,
  subdir: 'coxdev'
)

# The C++ headers (header-only: coxdev.h includes coxdev_impl.h), for
# packages that compile against coxdev: see coxdev.get_include()
py.install_sources(
  ['cpp/include/coxdev.h', 'cpp/include/coxdev_impl.h'],
  subdir: 'coxdev/include'
)

# Extension module
py.extension_module('coxc',
  ['python/coxdev_py.cpp'],
  dependencies: pybind11_dep,
  include_directories: inc,
  cpp_args: ['-DPY_INTERFACE=1'],
  install: true,
  subdir: 'coxdev'
)
