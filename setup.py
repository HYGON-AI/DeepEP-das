import os
import subprocess
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.0.0' + revision,
        packages=setuptools.find_packages(include=['deep_ep']),
        include_package_data=True,
        package_data={"deep_ep": ["deep_ep_cpp.cpython-310-x86_64-linux-gnu.so"]},
        zip_safe=False,
    )
