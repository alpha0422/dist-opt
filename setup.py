import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='distopt',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='distopt',
            include_dirs=[os.path.dirname(os.path.realpath(__file__))],
            sources=['csrc/dist_opt.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    test_suite="tests",
)
