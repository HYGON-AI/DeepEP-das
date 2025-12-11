import os, sys
import subprocess
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from datetime import datetime

date_tag = datetime.now().strftime("%Y%m%d")

# -----------------------
# 解析自定义参数
# -----------------------
shmem = None
other = []
for arg in sys.argv:
    if arg.startswith("--shmem="):
        shmem = arg.split("=", 1)[1]
        if shmem == "rocm":
            shmem = "a"
        elif shmem == "nv":
            shmem = "b"
    else:
        other.append(arg)

sys.argv = other

if __name__ == '__main__':
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.0.0' + revision + shmem + '.' + date_tag,
        packages=setuptools.find_packages(include=['deep_ep']),
        include_package_data=True,
        package_data={"deep_ep": ["deep_ep_cpp.cpython-310-x86_64-linux-gnu.so"]},
        zip_safe=False,
    )
