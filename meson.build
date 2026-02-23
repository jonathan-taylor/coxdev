project('coxdev', 'cpp',
  version: run_command('python', '-c', 'import setuptools_scm; print(setuptools_scm.get_version())', check: true).stdout().strip(),
  default_options: ['cpp_std=c++17', 'warning_level=3'])

py = import('python').find_installation(pure: false)

# Eigen is vendored in src/eigen
inc = include_directories('eigen', 'R_pkg/coxdev/inst/include')

# Pybind11 dependency
pybind11_dep = dependency('pybind11', required: false)

if not pybind11_dep.found()
    pybind11_incdir = run_command(py,
      ['-c', 'import pybind11; print(pybind11.get_include())'],
      check: true
    ).stdout().strip()
    pybind11_dep = declare_dependency(include_directories: include_directories(pybind11_incdir))
endif


# Includes
eigen_inc = include_directories('eigen')
coxdev_inc = include_directories('R_pkg/coxdev/inst/include')

python_sources = [
  'coxdev/__init__.py',
  'coxdev/base.py',
  'coxdev/stratified.py'
]

py.install_sources(
  python_sources,
  subdir: 'coxdev'
)

# Extension module
py.extension_module('coxc',
  'R_pkg/coxdev/src/coxdev.cpp',
  dependencies: pybind11_dep,
  include_directories: inc,
  cpp_args: ['-DPY_INTERFACE=1'],
  install: true,
  subdir: 'coxdev'
)

# Install Python sources
# install_subdir(
#     'coxdev',
#     install_dir: py.get_install_dir(),
#     exclude_directories: ['__pycache__', '.ipynb_checkpoints'],
#     exclude_files: ['.gitignore']
# )
