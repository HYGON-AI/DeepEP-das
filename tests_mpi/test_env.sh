#!/bin/bash
# =============================================================================
# DeepEP + RCCL/NCCL 环境配置
# 适用于: mpirun 启动的多节点训练
# 网络: InfiniBand (SHCA) 或 RoCE
# =============================================================================
export PYTHONPATH=$(pwd)

# rocSHMEM
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_TOPO_FILE_FORCE=$(pwd)/tests_mpi/topo.config
# NMZ使用
# export ROCSHMEM_DISABLE_HDP_FLUSH=1
# export ROCSHMEM_GDR_DISABLE_XDP=1

# # duSHMEM
# export LD_LIBRARY_PATH=/opt/dtk/dushmem/lib:$LD_LIBRARY_PATH
# export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_2:1,1:mlx5_3:1,2:mlx5_4:1,3:mlx5_5:1,4:mlx5_6:1,5:mlx5_7:1,6:mlx5_8:1,7:mlx5_9:1
# export NVSHMEM_SYMMETRIC_SIZE=10737418240
