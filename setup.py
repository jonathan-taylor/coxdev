import os
import shutil

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')
import setuptools

from setuptools import setup, Extension
import versioneer
import pybind11
#from cythexts import cyproc_exts, get_pyx_sdist

# Get various parameters for this version, stored in ISLP/info.py

class Bunch(object):
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith('__'):
                continue
            self.__dict__[key] = name

def read_vars_from(ver_file):
    """ Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    # Use exec for compabibility with Python 3
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)

info = read_vars_from(os.path.join('coxdev', 'info.py'))

# get long_description

long_description = open('README.md', 'rt', encoding='utf-8').read()
long_description_content_type = 'text/markdown'

# find eigen source directory of submodule

dirname = os.path.abspath(os.path.dirname(__file__))
eigendir = os.path.abspath(os.path.join(dirname, 'eigen'))

if 'EIGEN_LIBRARY_PATH' in os.environ:
    eigendir = os.path.abspath(os.environ['EIGEN_LIBRARY_PATH'])

# Ensure Eigen headers are available
def ensure_eigen_available():
    """Ensure Eigen headers are available for compilation"""
    if not os.path.exists(eigendir):
        raise RuntimeError(
            f"Eigen directory not found at {eigendir}. "
            "Please ensure the Eigen submodule is initialized: "
            "git submodule update --init --recursive"
        )
    
    # Check if Eigen headers are present
    eigen_headers = os.path.join(eigendir, 'Eigen')
    if not os.path.exists(eigen_headers):
        raise RuntimeError(
            f"Eigen headers not found at {eigen_headers}. "
            "Please ensure the Eigen submodule is properly initialized"
        )
    
    print(f"Using Eigen headers from: {eigendir}")

# Cox extension

EXTS=[Extension(
    'coxdev.coxc',
    sources=['R_pkg/coxdev/src/coxdev.cpp',
             f'R_pkg/coxdev/src/coxdev_strata.cpp'][:1],
    include_dirs=[pybind11.get_include(),
                  eigendir,
                  "R_pkg/coxdev/inst/include"],
    depends=["R_pkg/coxdev/inst/include/coxdev.h",
             "R_pkg/coxdev/inst/include/coxdev_strata.h"][:1],
    language='c++',
    extra_compile_args=['-std=c++17', '-DPY_INTERFACE=1'])]

cmdclass = versioneer.get_cmdclass()

def main(**extra_args):
    
    # All metadata is now handled by pyproject.toml
    # But we still need version from versioneer
    setup(
        version=versioneer.get_version(),
        packages=['coxdev'],
        ext_modules=EXTS,
        cmdclass=cmdclass,
        **extra_args
    )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()


