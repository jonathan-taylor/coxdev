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

# Cox extension

EXTS=[Extension(
    'coxc',
    sources=[f'src/coxdev.cpp'],
    include_dirs=[pybind11.get_include(),
                  eigendir],
    language='c++',
    extra_compile_args=['-std=c++17', '-DPY_INTERFACE=1'])]

# The cox extension shares the same C++ source as R and
# we will copy it to the src directory before setup.
# That way we have one true source. Should probably also add
# src/coxdev.cpp and src/coxdev.h to .gitignore

# Do some tasks before build
def do_prebuild_tasks():
    # copy the C++ source from R package
    src_files = ['R_pkg/coxdev/src/coxdev.cpp', 'R_pkg/coxdev/inst/include/coxdev.h']
    dest = 'src'
    if not os.path.exists(dest):
        os.makedirs(dest)
    for f in src_files:
        shutil.copy(f, dest)

cmdclass = versioneer.get_cmdclass()

def main(**extra_args):
    do_prebuild_tasks()
    setup(name=info.NAME,
          maintainer=info.MAINTAINER,
          maintainer_email=info.MAINTAINER_EMAIL,
          description=info.DESCRIPTION,
          url=info.URL,
          download_url=info.DOWNLOAD_URL,
          license=info.LICENSE,
          classifiers=info.CLASSIFIERS,
          author=info.AUTHOR,
          author_email=info.AUTHOR_EMAIL,
          platforms=info.PLATFORMS,
          version=versioneer.get_version(),
          packages = ['coxdev'],
          ext_modules = EXTS,
          include_package_data=True,
          data_files=[],
          scripts=[],
          long_description=long_description,
          long_description_content_type=long_description_content_type,
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()


