#!/bin/bash
# set -eux

export amd_comgr_DIR=${ROCM_PATH}/lib64/cmake
llvm15_path=${ROCM_PATH}/llvm/lib/clang/15.0.0
llvm17_path=${ROCM_PATH}/llvm/lib/clang/17.0.0
llvm18_path=${ROCM_PATH}/llvm/lib/clang/18

if [ -d "${llvm15_path}" ]; then
    echo "llvm version is 15.0.0"
    llvm_path=${llvm15_path}
fi
if [ -d "${llvm17_path}" ]; then
    echo "llvm version is 17.0.0"
    llvm_path=${llvm17_path}
fi
if [ -d "${llvm18_path}" ]; then
    echo "llvm version is 18"
    llvm_path=${llvm18_path}
fi

src_path=$(dirname "$(realpath $0)")

if [ ! -d "build_" ]; then
    mkdir -p build_
fi

PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYTHON_PLATLIB=$(python3 -c "from sysconfig import get_paths; print(get_paths()['platlib'])")

USE_NVSHMEM=OFF
USE_ROCSHMEM=OFF
ROCM_DISABLE_CTX=OFF
ROCM_DISABLE_MULTIQP=OFF
# 解析命令行参数
for arg in "$@"; do
    case $arg in
        rocshmem)
            USE_ROCSHMEM=ON
            ;;
        nvshmem|dushmem)
            USE_NVSHMEM=ON
            ;;
        ROCM_DISABLE_CTX=ON)
            ROCM_DISABLE_CTX=ON
            ;;
        ROCM_DISABLE_MULTIQP=ON)
            ROCM_DISABLE_MULTIQP=ON
            ;;
        *)
            echo "Usage: ./build.sh rocshmem [ROCM_DISABLE_CTX=ON] [ROCM_DISABLE_MULTIQP=ON] / ./build.sh dushmem"
            exit 1
            ;;
    esac
done

detect_offload_arch() {
    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        arch=$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx[0-9]+' | sort -r | head -n1)
        if [ -n "$arch" ]; then
            echo "--offload-arch=$arch"
            return 0
        fi
    fi
}
DETECTED_ARCH=$(detect_offload_arch)
echo "Current $DETECTED_ARCH"

echo "USE_NVSHMEM=$USE_NVSHMEM"
echo "USE_ROCSHMEM=$USE_ROCSHMEM"
echo "ROCM_DISABLE_CTX=$ROCM_DISABLE_CTX"
echo "ROCM_DISABLE_MULTIQP=$ROCM_DISABLE_MULTIQP"

# -------------------------- With rocSHMEM -------------------------- #
build_rocshmem()
{
    cd third-party/rocshmem/
    if [ ! -d "build" ]; then
        mkdir -p build
    fi
    cd build || {
        echo "错误: 无法进入构建目录 '$build_dir'"
        cd "$src_path"
        return 1
    }
    echo "cd third-party/rocshmem/build"
    bash ../scripts/build_configs/gda_mlx5
    echo "编译rocshmem成功"
    cd "$src_path"
}

if [ "$USE_ROCSHMEM" == "ON" ]; then
    if [ ! -d "third-party/rocshmem/src/" ]; then
        echo "download submodule..."
        git submodule update --init third-party/rocshmem
    fi

    if [ ! -d "third-party/rocshmem_install" ]; then
        mkdir -p third-party/rocshmem_install
    fi

    build_rocshmem
    SHMEM_INSTALL_PREFIX=$(pwd)/third-party/rocshmem_install
    COMPILE_OPTIONS=${COMPILE_OPTIONS:= -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1 -O3 -fgpu-rdc -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=deep_ep_cpp -D_GLIBCXX_USE_CXX11_ABI=1 ${DETECTED_ARCH}  -std=c++17 -Wno-return-type}
    if [ "$ROCM_DISABLE_CTX" == "ON" ]; then
        COMPILE_OPTIONS="-DROCM_DISABLE_CTX $COMPILE_OPTIONS"
    fi
    if [ "$ROCM_DISABLE_MULTIQP" == "ON" ]; then
        COMPILE_OPTIONS="-DROCM_DISABLE_MULTIQP $COMPILE_OPTIONS"
    fi
    SHMEM_LINK_OPTIONS=${SHMEM_LINK_OPTIONS:="-Wl,-rpath,${SHMEM_INSTALL_PREFIX}/lib/ -l:librocshmem.a"}
fi
# -------------------------- rocSHMEM END -------------------------- #
# -------------------------- With duSHMEM -------------------------- #
build_dushmem()
{
    cd third-party/dushmem-hip/
    source env.build.sh
    export CMAKE_PREFIX_PATH=${ROCM_PATH}/lib/cmake/amd_comgr:${ROCM_PATH}/lib64/cmake/amd_comgr:${CMAKE_PREFIX_PATH:-}
    export NVSHMEM_PREFIX=$src_path/third-party/dushmem_install
    if [ ! -d "build" ]; then
        mkdir -p build
    fi
    cd build || {
        echo "错误: 无法进入构建目录 '$build_dir'"
        cd "$src_path"
        return 1
    }
    echo "cd third-party/dushmem-hip/build"
    cmake ../
    make -j64
    make install
    echo "编译dushmem-hip成功"
    cd "$src_path"
}
if [ "$USE_NVSHMEM" == "ON" ]; then
    # if [ ! -d "third-party/dushmem-hip/src/" ]; then
    #     echo "download submodule..."
    #     git submodule update --init third-party/dushmem-hip
    # fi

    # if [ ! -d "third-party/dushmem_install" ]; then
    #     mkdir -p third-party/dushmem_install
    # fi
    # build_dushmem
    # SHMEM_INSTALL_PREFIX=$(pwd)/third-party/dushmem_install
    SHMEM_INSTALL_PREFIX=${ROCM_PATH}/dushmem
    COMPILE_OPTIONS=${COMPILE_OPTIONS:= -fPIC -DFORCE_DUSHMEM_API -DHIP_ENABLE_WARP_SYNC_BUILTINS -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1 -O3 -fgpu-rdc -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=deep_ep_cpp -D_GLIBCXX_USE_CXX11_ABI=1 ${DETECTED_ARCH} -std=c++17 -Wno-return-type}
    SHMEM_LINK_OPTIONS="-Wl,-rpath,${SHMEM_INSTALL_PREFIX}/lib/ -l:libdushmem_device.a -ldushmem_host"
fi
# -------------------------- duSHMEM END -------------------------- #

INCLUDE_PATHS=${INCLUDE_PATHS:=-Icsrc/ -I${SHMEM_INSTALL_PREFIX}/include/ -I/opt/mpi/include -I${PYTHON_PLATLIB}/torch/include -I${PYTHON_PLATLIB}/torch/include/torch/csrc/api/include -I${PYTHON_PLATLIB}/torch/include/TH -I${PYTHON_PLATLIB}/torch/include/THC -I${PYTHON_PLATLIB}/torch/include/THH -I/opt/dtk/include -I${PYTHON_INCLUDE}}

# 定义源文件列表（相对路径）
SOURCES=(
    "csrc/kernels/runtime.cu"
    "csrc/kernels/layout.cu"
    "csrc/kernels/intranode.cu"
    "csrc/kernels/internode.cu"
    "csrc/kernels/internode_ll.cu"
    "csrc/deep_ep.cu"
)

# 初始化对象文件列表
OBJECTS=()

# 检查是否需要强制重新编译（如果 shmem 库有更新）
FORCE_REBUILD=true

# 编译每个源文件
for src in "${SOURCES[@]}"; do
    # 生成对应的 .o 文件名（保留目录结构或扁平化）
    obj="build_/$(basename "${src%.cu}.o")"
    OBJECTS+=("$obj")

    # 检查是否需要重新编译
    if [[ "$FORCE_REBUILD" == true ]] || [[ ! -f "$obj" ]] || [[ "$src" -nt "$obj" ]]; then
        echo "Compiling $src -> $obj"
        hipcc ${INCLUDE_PATHS} -c "$src" -o "$obj" ${COMPILE_OPTIONS}
    else
        echo "Skipping $src (up to date)"
    fi
done

# 链接阶段
OUTPUT="deep_ep/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so"

# 检查是否需要重新链接
need_link=false
if [[ "$FORCE_REBUILD" == true ]] || [[ ! -f "$OUTPUT" ]]; then
    need_link=true
else
    for obj in "${OBJECTS[@]}"; do
        if [[ "$obj" -nt "$OUTPUT" ]]; then
            need_link=true
            break
        fi
    done
fi

if [[ "$need_link" == true ]]; then
    echo "Linking -> $OUTPUT"
    hipcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -shared -Wl,-O1 -Wl,-Bsymbolic-functions "${OBJECTS[@]}" -L${SHMEM_INSTALL_PREFIX}/lib/ -L/opt/mpi/lib -L/opt/dtk/hip/lib -L/usr/lib/x86_64-linux-gnu -lhipblaslt -lamdhip64 -o "$OUTPUT" -Wl,-rpath,/opt/dtk/lib -fgpu-rdc --hip-link ${DETECTED_ARCH} -shared -Wl,-soname,"$(basename "$OUTPUT")" -L"${llvm_path}/include/../lib/linux" -lclang_rt.builtins-x86_64 /opt/dtk/hip/lib/libgalaxyhip.so ${llvm_path}/lib/linux/libclang_rt.builtins-x86_64.a /opt/hyhal/lib/libhsa-runtime64.so -L${PYTHON_PLATLIB}/torch/lib -L/opt/dtk/lib -L/opt/dtk/hip/lib -L/usr/local/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -lamdhip64 -lc10_hip -ltorch_hip -lrocm-core -lrocm_smi64 ${SHMEM_LINK_OPTIONS} -fgpu-rdc --hip-link -lamdhip64 -lhsa-runtime64 -l:libmpi.so -Wl,-rpath,/opt/mpi/lib/ -libverbs -lmlx5
    echo "Successfully built $OUTPUT"
else
    echo "Skipping linking ($OUTPUT is up to date)"
fi

# build whl
echo "Using Python: $(which python3)"
python3 --version
if [ "$USE_NVSHMEM" == "ON" ]; then
    python setup.py bdist_wheel --shmem=nv
fi
if [ "$USE_ROCSHMEM" == "ON" ]; then
    python setup.py bdist_wheel --shmem=rocm
fi
echo "✅ Build complete:"
ls -lh dist/
