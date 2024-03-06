import os
from setuptools import setup

from torch.cuda import is_available as is_cuda_available
from torch.utils import cpp_extension

ext_name = "lcstorch"
version = "0.1"

# TODO define dependencies: torch, build: ninja, test: pytest-cov, pytest-xdist


def check_env_flag(name: str, default: str = '') -> bool:
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


extra_compile_args = []
extra_link_args = []
if check_env_flag('DEBUG'):
    extra_compile_args += ['-O0', '-g', '-DDEBUG']
    extra_link_args += ['-O0', '-g', '-DDEBUG']

# CPU + CUDA
if is_cuda_available():
    ext_modules = [cpp_extension.CUDAExtension(name=ext_name,
                                               sources=[
                                                   'cpu/lcs_cpu.cpp',
                                                   'cuda/lcs_cuda.cpp',
                                                   'cuda/lcs_cuda_kernel.cu',
                                                   'lcs.cpp'
                                               ],
                                               extra_compile_args=extra_compile_args,
                                               extra_link_args=extra_link_args)]
# CPU only
else:
    ext_modules = [cpp_extension.CppExtension(name=ext_name,
                                              sources=[
                                                  'lcs.cpp'
                                              ],
                                              extra_compile_args=extra_compile_args,
                                              extra_link_args=extra_link_args)]

setup(
    name=ext_name,
    ext_modules=ext_modules,
    version=version,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
