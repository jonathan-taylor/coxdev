"""
coxdev.get_include(): the installed C++ headers, and a program compiled
against them, as a package that builds on coxdev's C++ would be.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

import coxdev
from coxdev import CoxDeviance

EVENT = [3, 6, 8, 4, 6, 4, 3, 2, 2, 5, 3, 4]
STATUS = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]

# The implementation in one source file, the classes used from another, as in
# a package that compiles against coxdev.
IMPL = '#include <coxdev_impl.h>\n'
MAIN = '''
#include <coxdev.h>
#include <cstdio>
int main() {
  int n = 12;
  double event[] = {%s};
  int status[] = {%s};
  Eigen::VectorXd ev(n), start(n), w = Eigen::VectorXd::Ones(n), eta(n);
  Eigen::VectorXi st(n);
  for (int i = 0; i < n; ++i) {
    ev(i) = event[i];
    st(i) = status[i];
    start(i) = -std::numeric_limits<double>::infinity();
    eta(i) = -1.0 + 2.0 * i / (n - 1);
  }
  CoxDeviance cox(start, ev, st, w, true);
  double deviance = cox.compute_deviance(eta, w);
  std::printf("%%.17g\\n", deviance);
  for (int i = 0; i < n; ++i) std::printf("%%.17g\\n", cox.get_gradient()(i));
  return 0;
}
''' % (', '.join(str(float(e)) for e in EVENT), ', '.join(str(s) for s in STATUS))


def _eigen_include():
    # Eigen is not part of coxdev's wheel: a package compiling against coxdev
    # brings its own. Here, the repository's submodule, or COXDEV_EIGEN_INCLUDE.
    candidates = [os.environ.get('COXDEV_EIGEN_INCLUDE'),
                  os.path.join(os.path.dirname(__file__), '..', 'eigen')]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'Eigen', 'Dense')):
            return os.path.abspath(path)
    return None


def test_headers_installed():
    include = coxdev.get_include()
    assert os.path.isabs(include)
    for header in ['coxdev.h', 'coxdev_impl.h']:
        assert os.path.isfile(os.path.join(include, header))


def test_compile_against_headers(tmp_path):
    cxx = shutil.which(os.environ.get('CXX', 'c++'))
    eigen = _eigen_include()
    if cxx is None:
        pytest.skip('no C++ compiler')
    if eigen is None:
        pytest.skip('Eigen headers not found (set COXDEV_EIGEN_INCLUDE)')
    (tmp_path / 'impl.cpp').write_text(IMPL)
    (tmp_path / 'main.cpp').write_text(MAIN)
    exe = tmp_path / 'cox'
    subprocess.run([cxx, '-std=c++17', '-O1', '-I', coxdev.get_include(), '-I', eigen,
                    str(tmp_path / 'impl.cpp'), str(tmp_path / 'main.cpp'), '-o', str(exe)],
                   check=True, capture_output=True, text=True)
    out = np.array(subprocess.run([str(exe)], check=True, capture_output=True,
                                  text=True).stdout.split(), float)

    eta = np.linspace(-1, 1, len(EVENT))
    result = CoxDeviance(event=np.array(EVENT, float), status=np.array(STATUS))(eta)
    np.testing.assert_allclose(out[0], result.deviance, rtol=1e-12)
    np.testing.assert_allclose(out[1:], result.gradient, rtol=1e-12, atol=1e-14)
