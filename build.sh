#!/bin/bash
set -eux

if [ ! -d "build_" ]; then
    mkdir -p build_
fi

PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYTHON_PLATLIB=$(python3 -c "from sysconfig import get_paths; print(get_paths()['platlib'])")

# --------------------------------------------------------------------- #
USE_NVSHMEM=${USE_NVSHMEM:=OFF}
ROCSHMEM_INSTALL_PREFIX=${ROCSHMEM_INSTALL_PREFIX:=$(pwd)/rocshmem_dir}
COMPILE_OPTIONS=${COMPILE_OPTIONS:= -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1 -O3 -fgpu-rdc -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=deep_ep_cpp -D_GLIBCXX_USE_CXX11_ABI=1 --offload-arch=gfx936 -std=c++17 -Wno-return-type}
SHMEM_LINK_OPTIONS=${SHMEM_LINK_OPTIONS:="-Wl,-rpath,${ROCSHMEM_INSTALL_PREFIX}/lib/ -l:librocshmem.a"}

####
# 检查是否设置了USE_NVSHMEM环境变量
if [ "$USE_NVSHMEM" == "ON" ]; then
    COMPILE_OPTIONS+=" -DFORCE_NVSHMEM_API"
    ROCSHMEM_INSTALL_PREFIX=???/dushmem_dir
    SHMEM_LINK_OPTIONS="-Wl,-rpath,${ROCSHMEM_INSTALL_PREFIX}/lib/ -l:libnvshmem_device.a -lnvshmem_host"
fi

INCLUDE_PATHS=${INCLUDE_PATHS:=-Icsrc/ -I${ROCSHMEM_INSTALL_PREFIX}/include/ -I/opt/mpi/include -I${PYTHON_PLATLIB}/torch/include -I${PYTHON_PLATLIB}/torch/include/torch/csrc/api/include -I${PYTHON_PLATLIB}/torch/include/TH -I${PYTHON_PLATLIB}/torch/include/THC -I${PYTHON_PLATLIB}/torch/include/THH -I/opt/dtk/include -I${PYTHON_INCLUDE}}

hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/kernels/runtime.cu -o build_/runtime.o ${COMPILE_OPTIONS}
hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/kernels/layout.cu -o build_/layout.o ${COMPILE_OPTIONS}
hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/kernels/intranode.cu -o build_/intranode.o ${COMPILE_OPTIONS}
hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/kernels/internode.cu -o build_/internode.o ${COMPILE_OPTIONS}
hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/kernels/internode_ll.cu -o build_/internode_ll.o ${COMPILE_OPTIONS}
hipcc ${INCLUDE_PATHS} -c $(pwd)/csrc/deep_ep.cu -o build_/deep_ep.o ${COMPILE_OPTIONS}

hipcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -shared -Wl,-O1 -Wl,-Bsymbolic-functions build_/internode.o build_/intranode.o build_/runtime.o build_/deep_ep.o build_/layout.o build_/internode_ll.o -L${ROCSHMEM_INSTALL_PREFIX}/lib/ -L/opt/mpi/lib -L/opt/dtk/hip/lib -L/usr/lib/x86_64-linux-gnu -lhipblaslt -lamdhip64 -o deep_ep/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so -Wl,-rpath,/opt/dtk/lib -fgpu-rdc --hip-link --offload-arch=gfx936 -shared -Wl,-soname,deep_ep/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so -L"/opt/dtk/llvm/lib/clang/15.0.0/include/../lib/linux" -lclang_rt.builtins-x86_64 /opt/dtk/hip/lib/libgalaxyhip.so /opt/dtk/llvm/lib/clang/15.0.0/lib/linux/libclang_rt.builtins-x86_64.a /opt/hyhal/lib/libhsa-runtime64.so.1.11.0 -L${PYTHON_PLATLIB}/torch/lib -L/opt/dtk/lib -L/opt/dtk/hip/lib -L/usr/local/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -lamdhip64 -lc10_hip -ltorch_hip -lrocm-core -lrocm_smi64 ${SHMEM_LINK_OPTIONS} -fgpu-rdc --hip-link -lamdhip64 -lhsa-runtime64 -l:libmpi.so -Wl,-rpath,/opt/mpi/lib/ -libverbs -lmlx5

# build whl
echo "Using Python: $(which python3)"
python3 --version
python setup.py bdist_wheel
echo "✅ Build complete:"
ls -lh dist/
