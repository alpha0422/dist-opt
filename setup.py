from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='distopt',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='distopt', 
            sources=['csrc/dist_opt.cpp'], 
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    test_suite="tests",
)
