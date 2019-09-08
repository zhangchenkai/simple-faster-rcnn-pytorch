from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext

ext_modules = [Extension("_nms_gpu_post", ["_nms_gpu_post.pyx"])]
setup(
    name="Hello pyx",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)
