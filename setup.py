import os, sys
import subprocess
import setuptools
import subprocess
import sysconfig
from typing import Optional
import subprocess
from setuptools import Distribution

import torch

pwd = os.path.dirname(os.path.abspath(__file__))
add_git_version = False
if int(os.environ.get('ADD_GIT_VERSION', '0')) == 1:
    add_git_version = True

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

def get_version_add(sha: Optional[str] = None) -> str:
    command = "git config --global --add safe.directory " + pwd
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    deepep_root = os.path.dirname(os.path.abspath(__file__))
    add_version_path = os.path.join(os.path.join(deepep_root, "deep_ep"), "version.py")
    major, minor, _ = torch.__version__.split('.')
    if add_git_version:
        if sha != 'Unknown':
            if sha is None:
                sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=deepep_root).decode('ascii').strip()
            if (major, minor) >= ('2', '4'):
                version = 'das.opt1.' + sha[:7] + shmem
    else:
        if (major, minor) >= ('2', '4'):
            version = 'das.opt1'

    if os.getenv("ROCM_PATH"):
        rocm_path = os.getenv('ROCM_PATH', "")
        rocm_version_path = os.path.join(rocm_path, '.info', "rocm_version")
        with open(rocm_version_path, 'r',encoding='utf-8') as file:
            lines = file.readlines()
        rocm_version=lines[0].replace(".", "")
        version += ".dtk" + rocm_version

    new_version_content = f"""
try:
    __version__ = "1.0.0"
    __version_tuple__ = (1, 0, 0)
    __hcu_version__ = f'1.0.0+{version}'

    from deep_ep.version import __version__, __version_tuple__, __hcu_version__
except Exception as e:
    import warnings
    warnings.warn(f"Failed to read commit hash:\\n + str(e)",
                  RuntimeWarning, stacklevel=2)
    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)

def _prev_minor_version_was(version_str):
    '''Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version is 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    '''
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    # Note - this won't do the right thing when we release 1.0!
    # assert __version_tuple__[0] == 0
    assert isinstance(__version_tuple__[1], int)
    return version_str == f"{{__version_tuple__[0]}}.{{__version_tuple__[1] - 1}}"

def _prev_minor_version():
    '''For the purpose of testing, return a previous minor version number.'''
    # In dev tree, this will return "0.-1", but that will work fine"
    assert isinstance(__version_tuple__[1], int)
    return f"{{__version_tuple__[0]}}.{{__version_tuple__[1] - 1}}"
"""

    with open(add_version_path, encoding="utf-8",mode="w") as file:
        file.write(new_version_content)
    file.close()

def get_version():
    get_version_add()
    version_file = 'deep_ep/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__hcu_version__']

def get_deepep_version() -> str:
    version = get_version()
    return version

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

if __name__ == '__main__':
    setuptools.setup(
        name='deep_ep',
        version=get_deepep_version(),
        packages=setuptools.find_packages(include=['deep_ep']),
        include_package_data=True,
        package_data={"deep_ep": [f"deep_ep_cpp{sysconfig.get_config_var('EXT_SUFFIX')}"]},
        zip_safe=False,
        distclass=BinaryDistribution,
    )
