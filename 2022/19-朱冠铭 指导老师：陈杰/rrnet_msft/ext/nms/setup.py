# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy as np


def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_ASCEND():
    """Locate the CANN environment on the system

    Returns a dict with keys 'home', 'atc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the ASCENDHOME env variable. If not found, everything
    is based on finding 'atc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'ASCENDHOME' in os.environ:
        home = os.environ['ASCENDHOME']
        atc = pjoin(home, 'bin', 'atc')
    else:
        # otherwise, search the PATH for ATC
        default_path = pjoin(os.sep, 'usr', 'local', 'Ascend', 'bin')
        atc = find_in_path('atc', os.environ['PATH'] + os.pathsep + default_path)
        if atc is None:
            raise EnvironmentError('The atc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(atc))

    cudaconfig = {'home': home, 'atc': atc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


NPU = locate_ASCEND()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_atc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/atc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', NPU['atc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['atc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_atc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension('nms.gpu_nms',
              ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
              library_dirs=[NPU['lib64']],
              libraries=['cudart'],
              language='c++',
              runtime_library_dirs=[NPU['lib64']],
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with atc and not with
              # gcc the implementation of this trick is in customize_compiler() below
              extra_compile_args={'gcc': ["-Wno-unused-function"],
                                  'atc': ['-arch=sm_35',
                                           '--ptxas-options=-v',
                                           '-c',
                                           '--compiler-options',
                                           "'-fPIC'"]},
              include_dirs=[numpy_include, NPU['include']]
              )
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)
